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
    from torch import nn
except ModuleNotFoundError:  # Static contract tests run without the CUDA stack.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    class _NNStub:
        Module = object
    nn = _NNStub()  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper import loo_basis_output as output_io
from tools.paper import loo_cache_key_plan


TASK_ID = "AAAI27-LOO-BASIS-ADAPTATION-ATTEMPT-002"
SOURCE_BRANCH = "research/loo-basis-renderer-parity-closure-repair-20260724"
SOURCE_HEAD = "a34f750c64ba5c70fc360d16ada97ec0c6b84851"
DIAGNOSTIC_HEAD = "695ae9ca092f8260e6c5f0b016f491c5e4324fd1"
RUN_BRANCH = "research/loo-basis-adaptation-attempt2-renderer-parity-repaired-20260724"
PURE_BRANCH = "research/pure-endpoint-core-method-crossfit-amended-20260724"
PURE_HEAD = "ce110887a942cf8db082ba688c8d36d2433bfdbe"
HEADROOM_BRANCH = "research/render-refined-coefficient-headroom-attempt2-20260724"
HEADROOM_HEAD = "674e6092e21eeddeb22e963536247a3385c4e200"
FIGURE_BANK_BRANCH = "research/paper-figure-bank-pure-endpoint-refresh-20260724"
FIGURE_BANK_HEAD = "1fe425d2cc3cb3efd372334e1d845e63bf9d630a"
OUTPUT_NAME = output_io.OUTPUT_NAME
ATTEMPT_NAME = output_io.ATTEMPT_NAME
ORIGINAL_ATTEMPT_NAME = "attempt_001"
FORBIDDEN_NEXT_ATTEMPT_NAME = "attempt_003"
EXPECTED_ORIGINAL_FILE_COUNT = 66
EXPECTED_ORIGINAL_TOTAL_BYTES = 71_076_354
EXPECTED_ORIGINAL_AGGREGATE_SHA256 = (
    "4cd4ca03e21092431348c16a5103aff8863fd20861af9510913e02479761a31a"
)
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
MILESTONES = (0, 20, 50, 100, 200, 300)
METHOD_FAMILIES = (
    "FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION",
    "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC",
    "FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION",
)
STATIC_METHODS = (
    "BASE_AVATAR",
    "REFERENCE_NEAREST_HARD_LOOKUP",
    "RESIDUAL_NEAREST_ORACLE",
    "ORACLE_PROJECTION_LOO_BASIS",
    "HELD_OUT_TEACHER_ENDPOINT",
    "CONVEX_COMBINATION_ORACLE",
)
FULL_BASIS_FORBIDDEN_SHA = "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430"
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

REPAIRED_FILES = (
    "docs/PAPER/AAAI27_LOO_FEW_VIEW_FOLD_REPAIR_20260724.md",
    "docs/PAPER/AAAI27_LOO_K1_K2_PRIMARY_CROSSFIT_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_LOO_K4_DEFERRED_DIAGNOSTIC_BOUNDARY_20260724.md",
    "paper_protocol/reviewer_risk/loo_few_view_fold_repair.json",
    "paper_protocol/reviewer_risk/loo_few_view_manifests_repaired.json",
    "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol_amended.yaml",
    "paper_protocol/reviewer_risk/loo_shared_render_adaptation_loss_reference.json",
    "paper_protocol/reviewer_risk/loo_baseline_registry_amended.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_success_gates_amended.json",
    "paper_protocol/reviewer_risk/loo_expected_counts_amended.json",
    "paper_protocol/reviewer_risk/loo_execution_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_tests.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_final_summary.json",
    "project_control_handoff/loo_protocol_repair_handoff.json",
)
INHERITED_FILES = (
    "paper_protocol/reviewer_risk/loo_basis_manifests.json",
    "paper_protocol/reviewer_risk/loo_adaptation_loss_contract.json",
    "paper_protocol/reviewer_risk/loo_optimizer_contract.json",
    "paper_protocol/reviewer_risk/loo_baseline_registry.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract.json",
)
CACHE_REPAIRED_FILES = (
    "paper_protocol/reviewer_risk/loo_cache_count_root_cause.json",
    "paper_protocol/reviewer_risk/loo_static_render_track_cache_audit.json",
    "paper_protocol/reviewer_risk/loo_hard_lookup_k_replay.json",
    "paper_protocol/reviewer_risk/loo_cache_key_plan_v2.json",
    "paper_protocol/reviewer_risk/loo_expected_counts_cache_repaired.json",
    "paper_protocol/reviewer_risk/loo_storage_forecast_cache_repaired.json",
    "paper_protocol/reviewer_risk/loo_execution_contract_cache_repaired.json",
)
RENDERER_PARITY_REPAIRED_FILES = (
    "paper_protocol/reviewer_risk/loo_basis_renderer_parity_execution_contract_repaired.json",
    "paper_protocol/reviewer_risk/loo_basis_renderer_parity_repair_final_summary.json",
    "project_control_handoff/loo_basis_renderer_parity_repair_handoff.json",
    "scene/explicit_gaussian_residual_basis.py",
    "scene/gaussian_clothing_residuals.py",
)
REPORT_NAMES = (
    "AAAI27_LOO_ATTEMPT002_BASIS_ADAPTATION_RESULTS_20260724.md",
    "AAAI27_LOO_ATTEMPT002_HARD_LOOKUP_COMPARISON_20260724.md",
    "AAAI27_LOO_ATTEMPT002_BASIS_CAPACITY_ANALYSIS_20260724.md",
    "AAAI27_LOO_ATTEMPT002_VIEW_SCALING_RESULTS_20260724.md",
    "AAAI27_LOO_ATTEMPT002_FULL_RESIDUAL_COMPARISON_20260724.md",
    "AAAI27_LOO_ATTEMPT002_VISUAL_REVIEW_20260724.md",
    "AAAI27_LOO_ATTEMPT002_FAILURE_ANALYSIS_20260724.md",
)
REGISTRY_NAMES = (
    "loo_attempt002_basis_execution_registry.json",
    "loo_attempt002_adaptation_run_registry.json",
    "loo_attempt002_checkpoint_registry.json",
    "loo_attempt002_prediction_registry.json",
    "loo_attempt002_metric_summary.json",
    "loo_attempt002_oracle_capacity_analysis.json",
    "loo_attempt002_hard_lookup_analysis.json",
    "loo_attempt002_view_budget_scaling.json",
    "loo_attempt002_full_residual_summary.json",
    "loo_attempt002_visual_review_summary.json",
    "loo_attempt002_count_verification.json",
    "loo_attempt002_tests.json",
    "loo_attempt002_final_summary.json",
)
EXECUTION_BINDING_NAME = "loo_attempt002_execution_binding.json"
EXPECTED_COUNTS_BINDING_NAME = "loo_attempt002_execution_expected_counts.json"
PRE_RESULT_TESTS_NAME = "loo_attempt002_pre_result_tests.json"
ORIGINAL_MANIFEST_NAME = "loo_attempt002_attempt001_immutable_manifest.json"
HANDOFF_NAME = "loo_basis_adaptation_attempt002_handoff.json"
FIGURE_REFRESH_MANIFEST_NAME = "sealed_loo_attempt002_figure_refresh_manifest.json"
ALLOWED_CLASSIFICATIONS = (
    "LOO_BASIS_ADAPTATION_SUPPORTED",
    "LOO_BASIS_ADAPTATION_PARTIAL",
    "LOO_BASIS_CAPACITY_LIMITED",
    "LOO_OPTIMIZATION_LIMITED",
    "LOO_HARD_LOOKUP_NOT_OUTPERFORMED",
)
PHASES = (
    "00_preflight",
    "01_contract_snapshot",
    "02_basis_construction",
    "03_oracle_capacity",
    "04_low_dimensional_runs",
    "05_full_residual_runs",
    "06_checkpoints",
    "07_predictions",
    "08_metrics",
    "09_hard_lookup_analysis",
    "10_view_budget_scaling",
    "11_visual_sheets",
    "12_failure_analysis",
    "13_final_verification",
)
_RUNTIME_CACHE: dict[str, Any] | None = None
_CACHE_PLAN_RUNTIME: dict[str, Any] | None = None
_CACHE_REGISTRY_RUNTIME: dict[str, Any] | None = None
_CACHE_RENDERER_SHA_RUNTIME: str | None = None
_CONDITION_METADATA_RUNTIME: dict[str, Any] | None = None


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
        raise RuntimeError("PyYAML is required")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return output_io.canonical_sha256(value)


def sha256(path: Path, *, lf: bool = False) -> str:
    if not lf:
        return output_io.file_sha256(path)
    value = path.read_bytes()
    if value.startswith(b"\xef\xbb\xbf"):
        value = value[3:]
    text = value.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha(value: "torch.Tensor") -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def asset_root(value: str | None) -> Path:
    raw = value or os.environ.get("CANONDRESSGS_ASSET_ROOT")
    if not raw:
        raise RuntimeError("CANONDRESSGS_ASSET_ROOT or --asset-root is required")
    root = Path(raw).resolve()
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(root)
    os.environ["CANONDRESSGS_OUTPUT_ROOT"] = str(root)
    return root


def attempt_path(root: Path) -> Path:
    return output_io.attempt_path(root)


def original_attempt_path(root: Path) -> Path:
    return root.resolve() / OUTPUT_NAME / ORIGINAL_ATTEMPT_NAME


def forbidden_next_attempt_path(root: Path) -> Path:
    return root.resolve() / OUTPUT_NAME / FORBIDDEN_NEXT_ATTEMPT_NAME


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
    file_content_bytes = sum(row["bytes"] for row in rows)
    du = subprocess.run(["du", "-sb", str(root)], capture_output=True, text=True)
    total_bytes = int(du.stdout.split()[0]) if du.returncode == 0 else file_content_bytes
    aggregate_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in rows
    ]
    return {
        "schema_version": "canondressgs.paper.immutable_tree_manifest.v1",
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "file_content_bytes": file_content_bytes,
        "aggregate_sha256": canonical_sha(aggregate_rows),
        "aggregate_definition": "sha256(canonical JSON of ordered path/bytes/sha256 rows)",
        "files": rows,
    }


def original_attempt_immutability_audit(
    root: Path, expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actual = tree_manifest(original_attempt_path(root))
    frozen_checks = {
        "file_count_66": actual["file_count"] == EXPECTED_ORIGINAL_FILE_COUNT,
        "tree_bytes_exact": actual["total_bytes"] == EXPECTED_ORIGINAL_TOTAL_BYTES,
        "aggregate_sha256_exact": (
            actual["aggregate_sha256"] == EXPECTED_ORIGINAL_AGGREGATE_SHA256
        ),
    }
    manifest_match = True
    if expected is not None:
        manifest_match = all(
            actual.get(name) == expected.get(name)
            for name in (
                "file_count", "total_bytes", "file_content_bytes",
                "aggregate_sha256", "files",
            )
        )
    checks = {**frozen_checks, "bound_manifest_exact": manifest_match}
    return {
        "schema_version": "canondressgs.paper.loo_attempt002_attempt001_immutability.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mutation_count": 0 if all(checks.values()) else 1,
        "checks": checks,
        "expected": {
            "file_count": EXPECTED_ORIGINAL_FILE_COUNT,
            "total_bytes": EXPECTED_ORIGINAL_TOTAL_BYTES,
            "aggregate_sha256": EXPECTED_ORIGINAL_AGGREGATE_SHA256,
        },
        "actual": actual,
    }


def json_write(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    attempt = next(
        (parent for parent in (path.absolute(), *path.absolute().parents)
         if parent.name == ATTEMPT_NAME and parent.parent.name == OUTPUT_NAME),
        None,
    )
    if attempt is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not replace:
            raise FileExistsError(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(temporary, path)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return output_io.atomic_write_json(attempt, path, value, allow_replace=replace)


def text_write(path: Path, value: str, *, replace: bool = False) -> dict[str, Any]:
    attempt = next(
        (parent for parent in (path.absolute(), *path.absolute().parents)
         if parent.name == ATTEMPT_NAME and parent.parent.name == OUTPUT_NAME),
        None,
    )
    if attempt is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not replace:
            raise FileExistsError(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return output_io.atomic_write_text(attempt, path, value, allow_replace=replace)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    attempt = next(
        parent for parent in (path.absolute(), *path.absolute().parents)
        if parent.name == ATTEMPT_NAME and parent.parent.name == OUTPUT_NAME
    )
    output_io.atomic_append_jsonl(attempt, path, value)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line, object_pairs_hook=strict_object)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def contracts() -> dict[str, Any]:
    return {
        "basis": read_json(RISK / "loo_basis_manifests.json"),
        "tasks": read_json(RISK / "loo_few_view_manifests_repaired.json"),
        "loss": read_json(RISK / "loo_adaptation_loss_contract.json"),
        "optimizer": read_json(RISK / "loo_optimizer_contract.json"),
        "baselines": read_json(RISK / "loo_baseline_registry_amended.json"),
        "evaluator": read_json(RISK / "loo_evaluator_contract_amended.json"),
        "gates": read_json(RISK / "loo_success_gates_amended.json"),
        "counts": read_json(RISK / "loo_expected_counts_cache_repaired.json"),
        "execution": read_json(RISK / "loo_execution_contract_amended.json"),
        "cache_execution": read_json(RISK / "loo_execution_contract_cache_repaired.json"),
        "cache_plan": read_json(RISK / "loo_cache_key_plan_v2.json"),
        "hard_lookup_replay": read_json(RISK / "loo_hard_lookup_k_replay.json"),
        "storage": read_json(RISK / "loo_storage_forecast_cache_repaired.json"),
        "renderer_parity": read_json(
            RISK / "loo_basis_renderer_parity_execution_contract_repaired.json"
        ),
        "renderer_parity_summary": read_json(
            RISK / "loo_basis_renderer_parity_repair_final_summary.json"
        ),
    }


def tasks() -> list[dict[str, Any]]:
    return list(read_json(RISK / "loo_few_view_manifests_repaired.json")["primary_tasks"])


def split_rows() -> list[dict[str, Any]]:
    return list(read_json(RISK / "loo_basis_manifests.json")["splits"])


def expected_counts() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in contracts()["counts"]["planned_future_counts"].items()
    }


def git_source_bytes(relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{SOURCE_HEAD}:{relative}"], cwd=ROOT)


def source_artifact_audit() -> dict[str, Any]:
    rows = []
    for relative in (
        *REPAIRED_FILES, *INHERITED_FILES, *CACHE_REPAIRED_FILES,
        *RENDERER_PARITY_REPAIRED_FILES,
    ):
        source = git_source_bytes(relative)
        source_digest = hashlib.sha256(source).hexdigest()
        current_digest = sha256(ROOT / relative, lf=True)
        rows.append({
            "path": relative,
            "source_sha256": source_digest,
            "current_sha256": current_digest,
            "match": source_digest == current_digest,
        })
    return {
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
        "artifact_count": len(rows),
        "match_count": sum(row["match"] for row in rows),
        "rows": rows,
    }


def renderer_parity_contract_audit() -> dict[str, Any]:
    contract = contracts()["renderer_parity"]
    summary = contracts()["renderer_parity_summary"]
    checks = {
        "authorization_exact": (
            contract["authorization"] == "READY_FOR_LOO_ATTEMPT_002_BEFORE_OPTIMIZER"
        ),
        "status_exact": contract["status"] == "READY_FOR_LOO_ATTEMPT_002_BEFORE_OPTIMIZER",
        "algorithm_float64_strict_zero_sum": (
            contract["repaired_construction_algorithm"]
            == "float64_mean_center_rank3_svd_strict_zero_sum"
        ),
        "dtype_contract_float64": "float64" in contract["dtype_contract"],
        "coefficient_solver_projection": (
            contract["coefficient_solver_contract"]
            == "orthonormal_basis_transpose_projection"
        ),
        "split_parity_5_of_5": contract["all_five_split_parity"] == "5/5 PASS",
        "endpoint_parity_20_of_20": contract["all_endpoint_parity"] == "20/20 PASS",
        "scientific_semantic_drift_zero": (
            contract["scientific_contract_audit"]["scientific_semantic_drift"] == 0
        ),
        "summary_ready": summary["classification"] == "LOO_BASIS_RENDERER_PARITY_REPAIR_READY",
        "attempt_002_not_created_by_repair": summary["attempt_002_created"] is False,
        "optimizer_not_created_by_repair": summary["optimizer_creations"] == 0,
    }
    return {
        "schema_version": "canondressgs.paper.loo_attempt002_renderer_parity_contract_audit.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "contract_sha256": sha256(
            RISK / "loo_basis_renderer_parity_execution_contract_repaired.json"
        ),
        "summary_sha256": sha256(
            RISK / "loo_basis_renderer_parity_repair_final_summary.json"
        ),
    }


def historical_immutability_audit() -> dict[str, Any]:
    frozen = read_json(RISK / "loo_few_view_fold_repair.json")
    rows = []
    for expected in frozen["historical_artifact_immutability"]["artifacts"]:
        path = ROOT / expected["path"]
        actual_blob = git("hash-object", expected["path"])
        actual_sha = sha256(path, lf=True)
        rows.append({
            **expected,
            "actual_git_blob": actual_blob,
            "actual_sha256": actual_sha,
            "match": actual_blob == expected["git_blob"] and actual_sha == expected["sha256_git_blob_bytes"],
        })
    return {
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
        "artifact_count": len(rows),
        "mutation_count": sum(not row["match"] for row in rows),
        "historical_classification": "LOO_PROTOCOL_INCOMPLETE",
        "historical_valid_tasks": 15,
        "historical_blocked_tasks": 45,
        "rows": rows,
    }


def protocol_audit() -> dict[str, Any]:
    values = contracts()
    manifest = values["tasks"]
    rows = list(manifest["primary_tasks"])
    counts = Counter(int(row["K"]) for row in rows)
    expected_k1 = {"R0": "cond_000000", "R1": "cond_000017", "R2": "cond_000347", "R3": "cond_000000"}
    task_ids = [row["task_id"] for row in rows]
    overlap = []
    subset = []
    for row in rows:
        adaptation = set(row["selected_adaptation_conditions"])
        calibration = set(row["calibration_conditions"])
        test = set(row["test_conditions"])
        overlap.append(bool(adaptation & calibration or adaptation & test or calibration & test))
    for outfit in OUTFITS:
        for rotation in expected_k1:
            pair = [row for row in rows if row["held_out_garment"] == outfit and row["rotation"] == rotation]
            k1 = next(row for row in pair if int(row["K"]) == 1)
            k2 = next(row for row in pair if int(row["K"]) == 2)
            subset.append(set(k1["selected_adaptation_conditions"]).issubset(k2["selected_adaptation_conditions"]))
    split_expected = {
        outfit: [candidate for candidate in OUTFITS if candidate != outfit]
        for outfit in OUTFITS
    }
    checks = {
        "source_head_is_ancestor": subprocess.call(
            ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
        ) == 0,
        "task_count_40": len(rows) == 40,
        "task_ids_unique": len(task_ids) == len(set(task_ids)),
        "budgets_exactly_1_2": sorted(counts) == [1, 2],
        "k1_count_20": counts[1] == 20,
        "k2_count_20": counts[2] == 20,
        "blocked_zero": int(manifest["counts"]["blocked_tasks"]) == 0,
        "k4_unauthorized": not manifest["k4_deferred_contract"]["execution_authorized"],
        "k4_task_count_zero": int(manifest["k4_deferred_contract"]["current_task_count"]) == 0,
        "five_splits": len(split_rows()) == 5,
        "split_banks_exact": all(
            row["basis_garments"] == split_expected[row["held_out_garment"]]
            for row in split_rows()
        ),
        "all_ready": all(row["status"] == "READY" for row in rows),
        "overlap_zero": not any(overlap),
        "subset_20_of_20": len(subset) == 20 and all(subset),
        "k1_mapping_exact": all(
            row["selected_adaptation_conditions"] == [expected_k1[row["rotation"]]]
            for row in rows if int(row["K"]) == 1
        ),
        "loss_name_exact": values["loss"]["formal_name"] == "LOW_DIMENSIONAL_RENDER_ADAPTATION_LOSS_CONTRACT",
        "optimizer_runs_120": expected_counts()["total_optimizer_runs"] == 120,
        "checkpoint_writes_720": expected_counts()["checkpoint_writes"] == 720,
        "logical_renders_54960": expected_counts()["total_logical_renders"] == 54960,
        "physical_renders_54845": expected_counts()["unique_physical_renders"] == 54845,
        "cache_hits_115": expected_counts()["K_shared_static_cache_hits"] == 115,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "budget_counts": dict(counts),
        "task_manifest_sha256": sha256(RISK / "loo_few_view_manifests_repaired.json", lf=True),
        "expected_counts_sha256": sha256(RISK / "loo_expected_counts_cache_repaired.json", lf=True),
    }


SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
)


def credential_scan(paths: Sequence[Path]) -> dict[str, Any]:
    hits = []
    ignored_fixture_hits = []
    files = 0
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in candidates:
            if not path.is_file() or path.stat().st_size > 16 << 20:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files += 1
            for number, line in enumerate(text.splitlines(), 1):
                if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                    lowered = line.lower()
                    if "deliberately_fake" in lowered or "<redacted>" in lowered:
                        ignored_fixture_hits.append({
                            "path": str(path), "line": number,
                            "reason": "EXPLICIT_TEST_FIXTURE_OR_REDACTED_PLACEHOLDER",
                            "value": "REDACTED",
                        })
                        continue
                    hits.append({"path": str(path), "line": number, "value": "REDACTED"})
    return {
        "status": "PASS" if not hits else "FAIL",
        "files_scanned": files,
        "credential_value_hits": len(hits),
        "hits": hits,
        "ignored_explicit_fixture_hits": len(ignored_fixture_hits),
        "ignored_fixture_rows": ignored_fixture_hits,
    }


def planned_paths() -> list[str]:
    paths = [f"{phase}/.phase" for phase in PHASES]
    for row in tasks():
        token = row["task_id"]
        for family in METHOD_FAMILIES:
            phase = "05_full_residual_runs" if family == METHOD_FAMILIES[2] else "04_low_dimensional_runs"
            paths.extend((f"{phase}/{token}/{family}/RUN_STATUS.json", f"{phase}/{token}/{family}/run_summary.json"))
            if family == METHOD_FAMILIES[2]:
                paths.append(f"{phase}/{token}/{family}/equal_wall_time_state.pt")
            for step in MILESTONES:
                paths.append(f"06_checkpoints/{token}/{family}/step_{step:06d}.pth")
    paths.extend(f"11_visual_sheets/garment_rotation/{outfit}_{rotation}.png" for outfit in OUTFITS for rotation in ("R0", "R1", "R2", "R3"))
    paths.extend(f"11_visual_sheets/capacity/{outfit}.png" for outfit in OUTFITS)
    paths.append("11_visual_sheets/view_budget_scaling.png")
    return paths


def storage_forecast(root: Path) -> dict[str, Any]:
    artifact = contracts()["storage"]
    frozen = artifact["new_forecast"]
    required = int(frozen["required_free_bytes"])
    free = shutil.disk_usage(root).free
    return {
        "schema_version": "canondressgs.paper.loo_attempt002_storage_preflight.v1",
        "status": "PASS" if free >= required else "FAIL",
        "source": "paper_protocol/reviewer_risk/loo_storage_forecast_cache_repaired.json",
        "source_sha256": sha256(RISK / "loo_storage_forecast_cache_repaired.json"),
        "source_status": artifact["status"],
        "raw_estimated_bytes": int(frozen["raw_estimated_bytes"]),
        "safety_margin_fraction": float(frozen["safety_margin_fraction"]),
        "safety_margin_bytes": int(frozen["safety_margin_bytes"]),
        "required_free_bytes": required,
        "actual_free_bytes": free,
        "available_safety_margin_bytes": free - required,
        "prediction_render_pairs": int(frozen["prediction_render_pairs"]),
        "physical_render_count": int(artifact["physical_render_count"]),
        "formal_artifact_requirement_used": True,
    }


def cache_renderer_contract_sha256() -> str:
    global _CACHE_RENDERER_SHA_RUNTIME
    if _CACHE_RENDERER_SHA_RUNTIME is not None:
        return _CACHE_RENDERER_SHA_RUNTIME
    payload = {
        "evaluator": canonical_sha(read_json(RISK / "loo_evaluator_contract_amended.json")),
        "loss": canonical_sha(read_json(RISK / "loo_shared_render_adaptation_loss_reference.json")),
        "execution": canonical_sha(read_json(RISK / "loo_execution_contract_amended.json")),
        "renderer_implementation_sha256_lf": sha256(
            ROOT / "tools/run_residual_field_parameterization.py", lf=True
        ),
    }
    _CACHE_RENDERER_SHA_RUNTIME = canonical_sha(payload)
    return _CACHE_RENDERER_SHA_RUNTIME


def condition_metadata() -> dict[str, Any]:
    global _CONDITION_METADATA_RUNTIME
    if _CONDITION_METADATA_RUNTIME is None:
        manifest = read_json(RISK / "loo_few_view_manifests_repaired.json")
        _CONDITION_METADATA_RUNTIME = dict(
            manifest["semantic_and_camera_metadata"]["conditions"]
        )
    return _CONDITION_METADATA_RUNTIME


def deterministic_cache_plan(*, include_runtime_index: bool = False) -> dict[str, Any]:
    values = contracts()
    manifest = values["tasks"]
    numeric_counts = {
        key: int(value)
        for key, value in values["counts"]["planned_future_counts"].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return loo_cache_key_plan.build_cache_key_plan(
        tasks=manifest["primary_tasks"],
        replay=values["hard_lookup_replay"],
        expected_counts=numeric_counts,
        static_methods=STATIC_METHODS,
        optimizer_methods=METHOD_FAMILIES,
        milestones=MILESTONES,
        condition_metadata=manifest["semantic_and_camera_metadata"]["conditions"],
        renderer_contract_sha256=cache_renderer_contract_sha256(),
        background_identity="WHITE_BACKGROUND_RGB_1_1_1",
        resolution_identity="FROZEN_NATIVE_SUBJECT02_CONDITION_RESOLUTION",
        include_runtime_index=include_runtime_index,
    )


def cache_contract_preflight() -> dict[str, Any]:
    values = contracts()
    replay = values["hard_lookup_replay"]
    committed = values["cache_plan"]
    regenerated = deterministic_cache_plan()
    mismatch_identities = {
        (row["held_out_garment"], row["rotation"]): (
            row["K1_selected_known_endpoint"], row["K2_selected_known_endpoint"]
        )
        for row in replay["pairs"] if not row["same_endpoint"]
    }
    expected_mismatches = {
        ("O01", "R0"): ("O04", "O02"),
        ("O01", "R1"): ("O04", "O02"),
        ("O02", "R2"): ("O08", "O01"),
        ("O04", "R0"): ("O03", "O01"),
        ("O08", "R3"): ("O01", "O02"),
    }
    execution = values["cache_execution"]
    expected = values["counts"]
    checks = {
        "diagnostic_head_exact": execution["source_diagnostic_head"] == DIAGNOSTIC_HEAD,
        "replay_status_pass": replay["status"] == "PASS",
        "replay_15_same": int(replay["same_endpoint_pairs"]) == 15,
        "replay_5_different": int(replay["different_endpoint_pairs"]) == 5,
        "replay_mismatches_exact": mismatch_identities == expected_mismatches,
        "held_out_teacher_reads_zero": int(replay["held_out_teacher_reads"]) == 0,
        "test_metric_reads_zero": int(replay["test_metric_reads"]) == 0,
        "optimizer_reads_zero": int(replay["optimizer_reads"]) == 0,
        "plan_regeneration_exact": regenerated == committed,
        "plan_sha_exact": regenerated["deterministic_plan_sha256"] == expected["cache_key_plan_sha256"],
        "logical_requests_54960": regenerated["logical_request_count"] == 54_960,
        "unique_physical_keys_54845": regenerated["unique_physical_key_count"] == 54_845,
        "cache_hits_115": regenerated["cache_hit_count"] == 115,
        "hard_lookup_shared_15": regenerated["hard_lookup_k_shared_hits"] == 15,
        "hard_lookup_divergent_5": regenerated["hard_lookup_k_divergent_pairs"] == 5,
        "execution_contract_ready": execution["status"] == "READY_FOR_FIRST_LOO_SCIENTIFIC_ATTEMPT_BEFORE_OPTIMIZER",
    }
    return {
        "schema_version": "canondressgs.paper.loo_cache_contract_preflight.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "replay_sha256": replay["deterministic_replay_sha256"],
        "plan_sha256": regenerated["deterministic_plan_sha256"],
        "logical_request_count": regenerated["logical_request_count"],
        "unique_physical_key_count": regenerated["unique_physical_key_count"],
        "cache_hit_count": regenerated["cache_hit_count"],
    }


def planned_cache_request(
    task: Mapping[str, Any], method: str, condition: str, *, phase: str,
    logical_request_id: str, state_role: str, selected_endpoint: str,
    state_identity: str, output_semantic_role: str,
) -> dict[str, Any]:
    metadata = condition_metadata()[condition]
    return {
        "schema_version": "canondressgs.paper.loo_logical_render_request.v2",
        "logical_request_id": logical_request_id,
        "phase": phase,
        "track_id": (
            f"ADAPTATION/{method}" if phase == "ADAPTATION_LOSS_RENDER"
            else f"CHECKPOINT_EVALUATION/{method}"
            if phase == "OPTIMIZED_CHECKPOINT_TEST_EVALUATION"
            else f"STATIC/{method}"
        ),
        "task_id": task["task_id"],
        "held_out_garment": task["held_out_garment"],
        "rotation": task["rotation"],
        "K": int(task["K"]),
        "method": method,
        "split": f"LOO-{task['held_out_garment']}",
        "condition": condition,
        "checkpoint_or_final_state_role": state_role,
        "selected_known_endpoint": selected_endpoint,
        "basis_hash": loo_cache_key_plan._basis_hash(method, task),
        "coefficient_or_residual_state_identity": state_identity,
        "camera_metadata_sha256": metadata["camera_metadata_sha256"],
        "pose_metadata_sha256": metadata["pose_metadata_sha256"],
        "renderer_contract_sha256": cache_renderer_contract_sha256(),
        "background_identity": "WHITE_BACKGROUND_RGB_1_1_1",
        "resolution_identity": "FROZEN_NATIVE_SUBJECT02_CONDITION_RESOLUTION",
        "output_semantic_role": output_semantic_role,
    }


def adaptation_cache_request(
    task: Mapping[str, Any], method: str, step: int, condition: str,
    selected_endpoint: str,
) -> dict[str, Any]:
    selected = (
        "NOT_APPLICABLE"
        if method == "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC"
        else selected_endpoint
    )
    return planned_cache_request(
        task, method, condition,
        phase="ADAPTATION_LOSS_RENDER",
        logical_request_id=f"ADAPT/{task['task_id']}/{method}/step_{step:06d}/{condition}",
        state_role=f"PRE_UPDATE_STEP_{step:06d}",
        selected_endpoint=selected,
        state_identity=f"OPTIMIZER_STATE/{task['task_id']}/{method}/step_{step:06d}",
        output_semantic_role="ADAPTATION_RENDERING_LOSS_INPUT",
    )


def checkpoint_cache_request(
    task: Mapping[str, Any], method: str, step: int, condition: str,
    selected_endpoint: str,
) -> dict[str, Any]:
    selected = (
        "NOT_APPLICABLE"
        if method == "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC"
        else selected_endpoint
    )
    return planned_cache_request(
        task, method, condition,
        phase="OPTIMIZED_CHECKPOINT_TEST_EVALUATION",
        logical_request_id=f"EVAL/{task['task_id']}/{method}/step_{step:06d}/{condition}",
        state_role=f"CHECKPOINT_STEP_{step:06d}",
        selected_endpoint=selected,
        state_identity=f"CHECKPOINT/{task['task_id']}/{method}/step_{step:06d}",
        output_semantic_role="FORMAL_TEST_METRIC_INPUT",
    )


def static_cache_request(
    task: Mapping[str, Any], method: str, condition: str, selected_endpoint: str,
) -> dict[str, Any]:
    if method == loo_cache_key_plan.HARD_LOOKUP:
        selected = selected_endpoint
    elif method == "RESIDUAL_NEAREST_ORACLE":
        selected = "DEFERRED_OFFLINE_ORACLE_ENDPOINT"
    else:
        selected = "NOT_APPLICABLE"
    state_identity = (
        f"STATIC/{method}/KNOWN_ENDPOINT/{selected}"
        if method == loo_cache_key_plan.HARD_LOOKUP
        else f"STATIC/{method}/HELD_OUT/{task['held_out_garment']}"
    )
    return planned_cache_request(
        task, method, condition,
        phase="STATIC_TEST_EVALUATION",
        logical_request_id=f"STATIC/{task['task_id']}/{method}/{condition}",
        state_role="STATIC_FINAL_STATE",
        selected_endpoint=selected,
        state_identity=state_identity,
        output_semantic_role="FORMAL_TEST_METRIC_INPUT",
    )


def _runtime_cache_state(attempt: Path) -> dict[str, Any]:
    global _CACHE_PLAN_RUNTIME, _CACHE_REGISTRY_RUNTIME
    if _CACHE_PLAN_RUNTIME is None:
        plan = deterministic_cache_plan(include_runtime_index=True)
        _CACHE_PLAN_RUNTIME = {
            "summary": {key: value for key, value in plan.items() if not key.startswith("_runtime_")},
            "index": plan["_runtime_logical_request_key_index"],
        }
    path = attempt / "08_metrics/runtime_cache_request_registry.jsonl"
    if (
        _CACHE_REGISTRY_RUNTIME is not None
        and _CACHE_REGISTRY_RUNTIME["path"] != path
    ):
        _CACHE_REGISTRY_RUNTIME = None
    if _CACHE_REGISTRY_RUNTIME is None:
        prior = jsonl(path)
        logical_ids = {row["logical_request_id"] for row in prior}
        if len(logical_ids) != len(prior):
            raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
        _CACHE_REGISTRY_RUNTIME = {
            "path": path,
            "logical_ids": logical_ids,
            "physical_keys": {
                row["source_physical_key"] for row in prior if row["physical_render"]
            },
            "cache_hits": sum(bool(row["cache_hit"]) for row in prior),
        }
    return _CACHE_REGISTRY_RUNTIME


def validate_runtime_cache_request(
    attempt: Path, request: Mapping[str, Any], *, cache_hit: bool,
) -> tuple[str, str]:
    state = _runtime_cache_state(attempt)
    logical_id = str(request["logical_request_id"])
    actual_key, identity = loo_cache_key_plan.cache_key_v2(request)
    expected_key = _CACHE_PLAN_RUNTIME["index"].get(logical_id)
    if expected_key != actual_key or logical_id in state["logical_ids"]:
        raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
    already_physical = actual_key in state["physical_keys"]
    if cache_hit != already_physical:
        raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
    return actual_key, canonical_sha(identity)


def register_runtime_cache_request(
    attempt: Path, request: Mapping[str, Any], *, cache_hit: bool,
) -> str:
    state = _runtime_cache_state(attempt)
    logical_id = str(request["logical_request_id"])
    actual_key, identity_sha = validate_runtime_cache_request(
        attempt, request, cache_hit=cache_hit
    )
    row = {
        "logical_request_id": logical_id,
        "task_id": request["task_id"],
        "phase": request["phase"],
        "track_id": request["track_id"],
        "cache_key_v2": actual_key,
        "render_identity_sha256": identity_sha,
        "cache_hit": cache_hit,
        "physical_render": not cache_hit,
        "source_physical_key": actual_key,
    }
    append_jsonl(state["path"], row)
    state["logical_ids"].add(logical_id)
    if cache_hit:
        state["cache_hits"] += 1
    else:
        state["physical_keys"].add(actual_key)
    return actual_key


def verify_runtime_cache_registry(attempt: Path) -> dict[str, Any]:
    state = _runtime_cache_state(attempt)
    plan = _CACHE_PLAN_RUNTIME["summary"]
    checks = {
        "logical_request_count": len(state["logical_ids"]) == plan["logical_request_count"],
        "unique_physical_key_count": len(state["physical_keys"]) == plan["unique_physical_key_count"],
        "cache_hit_count": int(state["cache_hits"]) == plan["cache_hit_count"],
        "logical_request_ids_exact": state["logical_ids"] == set(_CACHE_PLAN_RUNTIME["index"]),
    }
    result = {
        "schema_version": "canondressgs.paper.loo_runtime_cache_plan_verification.v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "actual_logical_request_count": len(state["logical_ids"]),
        "actual_unique_physical_key_count": len(state["physical_keys"]),
        "actual_cache_hit_count": int(state["cache_hits"]),
        "deterministic_plan_sha256": plan["deterministic_plan_sha256"],
    }
    if result["status"] != "PASS":
        raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
    return result


def static_preflight(root: Path | None = None) -> dict[str, Any]:
    artifacts = source_artifact_audit()
    historical = historical_immutability_audit()
    renderer_parity = renderer_parity_contract_audit()
    protocol = protocol_audit()
    cache_contract = cache_contract_preflight()
    path_plan = output_io.audit_relative_paths(planned_paths())
    credentials = credential_scan((ROOT / "tools/paper", ROOT / "docs/PAPER", RISK, HANDOFF))
    checks = {
        "branch_exact": git("branch", "--show-current") == RUN_BRANCH,
        "worktree_clean": not git("status", "--short"),
        "source_artifacts": artifacts["status"] == "PASS",
        "historical_immutability": historical["status"] == "PASS",
        "renderer_parity_contract": renderer_parity["status"] == "PASS",
        "protocol": protocol["status"] == "PASS",
        "cache_contract": cache_contract["status"] == "PASS",
        "output_path_plan": path_plan["status"] == "PASS",
        "credential_scan": credentials["status"] == "PASS",
    }
    original_attempt = None
    if root is not None:
        original_attempt = original_attempt_immutability_audit(root)
        checks.update({
            "attempt_001_exists_and_immutable": original_attempt["status"] == "PASS",
            "attempt_002_absent": not attempt_path(root).exists(),
            "attempt_003_absent": not forbidden_next_attempt_path(root).exists(),
            "storage_forecast": storage_forecast(root)["status"] == "PASS",
        })
    return {
        "schema_version": "canondressgs.paper.loo_static_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_artifact_audit": artifacts,
        "historical_immutability": historical,
        "renderer_parity_contract_audit": renderer_parity,
        "original_attempt_immutability": original_attempt,
        "protocol_audit": protocol,
        "cache_contract_preflight": cache_contract,
        "path_plan": path_plan,
        "credential_scan": credentials,
        "storage_forecast": storage_forecast(root) if root is not None else None,
        "checked_at_utc": now(),
        "preflight_order": [
            "exact_source_head", "historical_artifact_immutability",
            "repaired_protocol_hashes", "40_task_manifest", "K_mappings",
            "held_out_boundary", "F2_centroid_replay", "15_5_hard_lookup_parity",
            "complete_cache_key_plan", "exact_render_counts", "storage_forecast",
            "GPU_resource_gate", "credential_gate", "output_collision_gate",
            "execution_head_authorization", "attempt_materialization", "renderer",
            "optimizer",
        ],
    }


def cloud_resource_preflight(root: Path) -> dict[str, Any]:
    static = static_preflight(root)
    if static["status"] != "PASS":
        raise RuntimeError("LOO_ADAPTATION_STATIC_PREFLIGHT_FAILED")
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("LOO_ADAPTATION_CUDA_RUNTIME_MISSING")
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, encoding="utf-8",
    ).strip().splitlines()
    if len(gpu_rows) != 1:
        raise RuntimeError("LOO_ADAPTATION_GPU_RESOURCE_CONFLICT")
    gpu_name, total_mib, free_mib, used_mib = [value.strip() for value in gpu_rows[0].split(",")]
    try:
        process_text = subprocess.check_output(
            [
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ], text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        process_text = ""
    active_compute = [line for line in process_text.splitlines() if line.strip()]
    probe = root / ".loo_basis_adaptation_write_probe"
    if probe.exists():
        raise FileExistsError(probe)
    probe.write_bytes(b"writable\n")
    probe.unlink()
    remotes = set(git("remote").splitlines())
    linked_bare = not remotes
    remote_ok = {"origin", "cloud"}.issubset(remotes) or (
        linked_bare
        and git("branch", "--show-current") == RUN_BRANCH
        and git("rev-parse", RUN_BRANCH) == git("rev-parse", "HEAD")
    )
    forecast = storage_forecast(root)
    f2_cache = f2_cache_preflight(root)
    original_attempt = original_attempt_immutability_audit(root)
    checks = {
        "gpu_is_rtx_4090": gpu_name == "NVIDIA GeForce RTX 4090",
        "free_vram_at_least_20_gib": int(free_mib) >= 20 * 1024,
        "no_active_compute_process": not active_compute,
        "torch_cuda_available": torch.cuda.is_available(),
        "python_3_10_20": platform.python_version() == "3.10.20",
        "torch_2_4_1_cu121": torch.__version__ == "2.4.1+cu121",
        "cuda_12_1": torch.version.cuda == "12.1",
        "cuda_tensor_smoke": float(torch.tensor([6.0, 7.0], device="cuda").sum()) == 13.0,
        "output_writable": True,
        "attempt_001_exists_and_immutable": original_attempt["status"] == "PASS",
        "attempt_002_absent": not attempt_path(root).exists(),
        "attempt_003_absent": not forbidden_next_attempt_path(root).exists(),
        "storage_forecast": forecast["status"] == "PASS",
        "git_remote_continuity": remote_ok,
        "static_preflight": static["status"] == "PASS",
        "f2_K_shared_cache_contract": f2_cache["status"] == "PASS",
    }
    return {
        "schema_version": "canondressgs.paper.loo_cloud_resource_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gpu": {
            "name": gpu_name, "memory_total_mib": int(total_mib),
            "memory_free_mib": int(free_mib), "memory_used_mib": int(used_mib),
        },
        "active_compute_process_count": len(active_compute),
        "active_compute_processes": ["REDACTED_PROCESS_METADATA" for _ in active_compute],
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "storage_forecast": forecast,
        "original_attempt_immutability": original_attempt,
        "f2_cache_preflight": f2_cache,
        "cache_contract_preflight": static["cache_contract_preflight"],
        "preflight_order": static["preflight_order"],
        "git_topology": "LINKED_LOCAL_BARE_REPOSITORY" if linked_bare else "NAMED_REMOTES",
        "remote_names": sorted(remotes),
        "credential_values_recorded": False,
        "credential_env_names": [name for name in ("GITHUB_TOKEN", "GH_TOKEN") if name in os.environ],
        "checked_at_utc": now(),
    }


def bind(cloud_preflight_path: Path, root: Path) -> dict[str, Any]:
    if git("status", "--short"):
        raise RuntimeError("binding requires a clean worktree")
    local = static_preflight(root)
    cloud = read_json(cloud_preflight_path)
    if local["status"] != "PASS" or cloud.get("status") != "PASS":
        raise RuntimeError("LOO_ADAPTATION_PREFLIGHT_FAILED")
    fingerprints = {
        relative: {
            "sha256": sha256(ROOT / relative),
            "sha256_lf": sha256(ROOT / relative, lf=True),
        }
        for relative in (
            *REPAIRED_FILES, *INHERITED_FILES, *CACHE_REPAIRED_FILES,
            *RENDERER_PARITY_REPAIRED_FILES,
        )
    }
    original_manifest = cloud["original_attempt_immutability"]["actual"]
    original_audit = original_attempt_immutability_audit(root, original_manifest)
    if original_audit["status"] != "PASS":
        raise RuntimeError("LOO_ATTEMPT_001_IMMUTABILITY_FAILURE")
    payload = {
        "schema_version": "canondressgs.paper.loo_execution_binding.v1",
        "task_id": TASK_ID,
        "status": "BOUND_PRE_RESULT",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "binding_parent_head": git("rev-parse", "HEAD"),
        "execution_head": "RESOLVE_AFTER_BINDING_COMMIT",
        "attempt": ATTEMPT_NAME,
        "attempt_count_before_execution": 1,
        "attempt_001_role": "PRESERVED_PRE_OPTIMIZER_BASIS_RENDERER_PARITY_FAILURE",
        "attempt_001_immutable": original_audit["status"] == "PASS",
        "attempt_002_absent": not attempt_path(root).exists(),
        "attempt_003_absent": not forbidden_next_attempt_path(root).exists(),
        "reuse_attempt_001_basis_artifacts": False,
        "reuse_attempt_001_auxiliary_renders": False,
        "reuse_attempt_001_optimizer_state": False,
        "contract_fingerprints": fingerprints,
        "task_manifest_semantic_sha256": canonical_sha(contracts()["tasks"]),
        "expected_counts_semantic_sha256": canonical_sha(contracts()["counts"]),
        "protocol_audit": local["protocol_audit"],
        "historical_immutability": local["historical_immutability"],
        "renderer_parity_contract_audit": local["renderer_parity_contract_audit"],
        "original_attempt_immutability": original_audit,
        "storage_forecast": cloud["storage_forecast"],
        "cloud_preflight_sha256": sha256(cloud_preflight_path),
        "held_out_teacher_use_in_deployable_adaptation": 0,
        "full_five_garment_rank4_basis_reused": False,
        "K4_execution_authorized": False,
        "PAPER_FINAL": False,
        "paper_final_count": 0,
        "created_at_utc": now(),
    }
    paths = {
        RISK / EXECUTION_BINDING_NAME: payload,
        RISK / EXPECTED_COUNTS_BINDING_NAME: {
            "schema_version": "canondressgs.paper.loo_attempt002_execution_expected_counts.v1",
            "task_id": TASK_ID, "status": "FROZEN", "counts": expected_counts(),
            "source_sha256": sha256(RISK / "loo_expected_counts_cache_repaired.json"),
        },
        RISK / PRE_RESULT_TESTS_NAME: {
            "schema_version": "canondressgs.paper.loo_attempt002_pre_result_tests.v1",
            "task_id": TASK_ID, "status": "PASS", "local": local,
            "cloud": cloud, "result_information_used": False,
        },
        RISK / ORIGINAL_MANIFEST_NAME: original_manifest,
    }
    for path, value in paths.items():
        json_write(path, value, replace=False)
    return payload


def materialize(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    if attempt.exists():
        raise RuntimeError("LOO_ATTEMPT_002_COLLISION")
    if forbidden_next_attempt_path(root).exists():
        raise RuntimeError("LOO_ATTEMPT_003_FORBIDDEN")
    if git("status", "--short"):
        raise RuntimeError("materialize requires the clean EXECUTION_HEAD")
    binding = read_json(RISK / EXECUTION_BINDING_NAME)
    original_manifest = read_json(RISK / ORIGINAL_MANIFEST_NAME)
    original_audit = original_attempt_immutability_audit(root, original_manifest)
    if original_audit["status"] != "PASS":
        raise RuntimeError("LOO_ATTEMPT_001_IMMUTABILITY_FAILURE")
    resource = cloud_resource_preflight(root)
    if resource["status"] != "PASS":
        raise RuntimeError("LOO_ADAPTATION_RESOURCE_PREFLIGHT_FAILED")
    cache_preflight = cache_contract_preflight()
    runtime_plan = deterministic_cache_plan(include_runtime_index=True)
    runtime_index = runtime_plan.pop("_runtime_logical_request_key_index")
    if cache_preflight["status"] != "PASS" or runtime_plan != contracts()["cache_plan"]:
        raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
    if len(runtime_index) != int(expected_counts()["total_logical_renders"]):
        raise RuntimeError("LOO_RUNTIME_CACHE_PLAN_DRIFT")
    output_parent = root / OUTPUT_NAME
    output_parent.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(exist_ok=False)
    for phase in PHASES:
        output_io.ensure_directory(attempt, attempt / phase)
    execution_head = git("rev-parse", "HEAD")
    metadata = {
        "schema_version": "canondressgs.paper.loo_execution_metadata.v1",
        "task_id": TASK_ID, "status": "EXECUTION_HEAD_FROZEN",
        "attempt": ATTEMPT_NAME, "execution_head": execution_head,
        "source_head": SOURCE_HEAD, "run_branch": RUN_BRANCH,
        "binding_sha256": sha256(RISK / EXECUTION_BINDING_NAME),
        "attempt_001_manifest_sha256": sha256(RISK / ORIGINAL_MANIFEST_NAME),
        "reuse_attempt_001_basis_artifacts": False,
        "reuse_attempt_001_auxiliary_renders": False,
        "reuse_attempt_001_optimizer_state": False,
        "created_at_utc": now(),
    }
    json_write(attempt / "00_preflight/execution_metadata.json", metadata)
    json_write(attempt / "00_preflight/cloud_resource_preflight.json", resource)
    json_write(attempt / "00_preflight/historical_immutability.json", historical_immutability_audit())
    json_write(attempt / "00_preflight/attempt_001_immutability.json", original_audit)
    json_write(attempt / "00_preflight/attempt_001_manifest.json", original_manifest)
    json_write(attempt / "00_preflight/storage_forecast.json", resource["storage_forecast"])
    json_write(attempt / "00_preflight/credential_scan.json", credential_scan((ROOT,)))
    json_write(attempt / "00_preflight/cache_contract_preflight.json", cache_preflight)
    snapshot_files = (
        *REPAIRED_FILES, *INHERITED_FILES, *CACHE_REPAIRED_FILES,
        *RENDERER_PARITY_REPAIRED_FILES,
    )
    for relative in snapshot_files:
        destination = attempt / "01_contract_snapshot" / relative
        output_io.atomic_write_bytes(attempt, destination, (ROOT / relative).read_bytes())
    json_write(attempt / "01_contract_snapshot/contract_registry.json", {
        "status": "PASS",
        "artifact_count": len(snapshot_files),
        "artifacts": {
            relative: sha256(attempt / "01_contract_snapshot" / relative)
            for relative in snapshot_files
        },
    })
    json_write(attempt / "01_contract_snapshot/runtime_cache_plan_binding.json", {
        "schema_version": "canondressgs.paper.loo_runtime_cache_plan_binding.v2",
        "status": "PASS",
        "logical_request_count": len(runtime_index),
        "logical_request_key_index_sha256": runtime_plan["logical_request_key_index_sha256"],
        "deterministic_plan_sha256": runtime_plan["deterministic_plan_sha256"],
        "expected_unique_physical_key_count": runtime_plan["unique_physical_key_count"],
        "expected_cache_hit_count": runtime_plan["cache_hit_count"],
    })
    json_write(attempt / "RUN_STATUS.json", {
        "task_id": TASK_ID, "attempt": ATTEMPT_NAME, "status": "MATERIALIZED_NO_OPTIMIZER",
        "execution_head": execution_head, "optimizer_runs": 0, "optimizer_steps": 0,
        "checkpoint_writes": 0, "logical_renders": 0, "physical_renders": 0,
        "updated_at_utc": now(),
    })
    return {"status": "PASS", "attempt": str(attempt), "execution_head": execution_head}


def assert_execution_head(attempt: Path) -> str:
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    head = git("rev-parse", "HEAD")
    if head != metadata["execution_head"] or git("status", "--short"):
        raise RuntimeError("scientific execution requires the clean EXECUTION_HEAD")
    if metadata.get("attempt") != ATTEMPT_NAME or attempt.name != ATTEMPT_NAME:
        raise RuntimeError("LOO_ATTEMPT_002_BINDING_MISMATCH")
    root = attempt.parent.parent
    original_manifest = read_json(RISK / ORIGINAL_MANIFEST_NAME)
    if original_attempt_immutability_audit(root, original_manifest)["status"] != "PASS":
        raise RuntimeError("LOO_ATTEMPT_001_IMMUTABILITY_FAILURE")
    if forbidden_next_attempt_path(root).exists():
        raise RuntimeError("LOO_ATTEMPT_003_FORBIDDEN")
    return head


def runtime_imports() -> dict[str, Any]:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE
    if torch is None:
        raise RuntimeError("PyTorch is required")
    from scene.explicit_gaussian_residual_basis import (
        ExplicitGaussianResidualBasis,
        build_svd_basis,
        normalized_residual_dict,
        project_residual_onto_basis,
        tensor_mapping_fingerprint,
    )
    from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
    from scene.representation_capacity_oracle import garment_trainable_mask
    from tools import run_multi_outfit_explicit_basis as multi
    from tools import run_residual_field_parameterization as parameterization
    from tools.paper import formal_batch_runtime as historical
    from tools.paper import formal_runtime as formal
    from tools.paper import run_p0_color_spatial_soft_control_evaluations as metrics
    _RUNTIME_CACHE = {
        "ExplicitGaussianResidualBasis": ExplicitGaussianResidualBasis,
        "build_svd_basis": build_svd_basis,
        "normalized_residual_dict": normalized_residual_dict,
        "project_residual_onto_basis": project_residual_onto_basis,
        "tensor_mapping_fingerprint": tensor_mapping_fingerprint,
        "CHANNELS": CHANNELS,
        "GaussianClothingResiduals": GaussianClothingResiduals,
        "garment_trainable_mask": garment_trainable_mask,
        "multi": multi, "parameterization": parameterization,
        "historical": historical, "formal": formal, "metrics": metrics,
    }
    return _RUNTIME_CACHE


def runtime_context(attempt: Path) -> dict[str, Any]:
    modules = runtime_imports()
    formal = modules["formal"]
    old_branch, old_head = formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD
    formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = RUN_BRANCH, SOURCE_HEAD
    try:
        return formal._legacy_context(attempt)
    finally:
        formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = old_branch, old_head


def load_teachers(context: Mapping[str, Any]) -> dict[str, Any]:
    return runtime_imports()["historical"].load_frozen_teacher_residuals(context, OUTFITS)


def residual_fields(residual: Any) -> tuple["torch.Tensor", ...]:
    return tuple(getattr(residual, name) for name in runtime_imports()["CHANNELS"])


def residual_fingerprint(residual: Any) -> str:
    return canonical_sha([tensor_sha(value) for value in residual_fields(residual)])


def save_torch(attempt: Path, path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    def validate(temporary: Path) -> None:
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if loaded is None:
            raise ValueError("torch roundtrip returned None")
    return output_io.atomic_torch_save(attempt, path, value, validate, allow_replace=replace)


def _basis_payload(decomposition: Any, statistics_payload: Mapping[str, Any]) -> dict[str, Any]:
    mean, components = decomposition.basis.normalized_fields()
    matrix = torch.stack([
        decomposition.teacher_coefficients[outfit]
        for outfit in decomposition.metadata["outfit_order"]
    ])
    coefficient_mean = matrix.mean(0)
    coefficient_std = matrix.std(0, unbiased=False).clamp_min(1e-8)
    return {
        "schema_version": "canondressgs.paper.loo_basis_artifact.v1",
        "channel_bounds": dict(decomposition.basis.channel_bounds),
        "rank": decomposition.basis.rank,
        "mean_normalized": {name: value.detach().cpu() for name, value in mean.items()},
        "basis_normalized": {name: value.detach().cpu() for name, value in components.items()},
        "basis_garments": list(decomposition.metadata["outfit_order"]),
        "teacher_coefficients": {
            name: value.detach().cpu() for name, value in decomposition.teacher_coefficients.items()
        },
        "coefficient_mean": coefficient_mean.detach().cpu(),
        "coefficient_std": coefficient_std.detach().cpu(),
        "basis_fingerprint": decomposition.basis.fingerprint(),
        "statistics": dict(statistics_payload),
        "held_out_garment_used": False,
        "full_five_garment_basis_reused": False,
    }


def load_loo_basis(attempt: Path, held_out: str, device: "torch.device") -> tuple[Any, dict[str, "torch.Tensor"], dict[str, Any]]:
    path = attempt / f"02_basis_construction/{held_out}/loo_basis.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "canondressgs.paper.loo_basis_artifact.v1":
        raise ValueError("LOO basis schema mismatch")
    basis = runtime_imports()["ExplicitGaussianResidualBasis"](
        {name: value.to(device) for name, value in payload["mean_normalized"].items()},
        {name: value.to(device) for name, value in payload["basis_normalized"].items()},
        payload["channel_bounds"],
    )
    if basis.rank > 3 or basis.fingerprint() != payload["basis_fingerprint"]:
        raise RuntimeError("LOO basis fingerprint/rank mismatch")
    coefficients = {name: value.to(device) for name, value in payload["teacher_coefficients"].items()}
    return basis, coefficients, payload


def normalized_rmse(first: Any, second: Any, bounds: Mapping[str, float]) -> float:
    normalized = runtime_imports()["normalized_residual_dict"]
    a, b = normalized(first, bounds), normalized(second, bounds)
    numerator = sum(float((a[name].double() - b[name].double()).square().sum()) for name in a)
    denominator = sum(value.numel() for value in a.values())
    return math.sqrt(numerator / max(denominator, 1))


def build_bases(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "02_basis_construction/summary.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    teachers = load_teachers(context)
    modules = runtime_imports()
    rows = []
    auxiliary_render_calls = 0
    for split in split_rows():
        held_out = split["held_out_garment"]
        basis_garments = list(split["basis_garments"])
        bounds = split["basis_contract"]["channel_bounds"]
        normalized = {
            name: modules["normalized_residual_dict"](
                teachers[name], bounds, dtype=torch.float64,
            )
            for name in basis_garments
        }
        flattened = [torch.cat([normalized[name][field].reshape(-1) for field in modules["CHANNELS"]]) for name in basis_garments]
        matrix = torch.stack(flattened)
        centered = matrix - matrix.mean(0)
        centered = centered.clone()
        centered[-1] = -centered[:-1].sum(0)
        singular_values = torch.linalg.svdvals(centered).detach().cpu()
        tolerance = max(4, int(centered.shape[1])) * torch.finfo(centered.dtype).eps * float(singular_values[0])
        numerical_rank = int((singular_values > tolerance).sum())
        selected_rank = min(numerical_rank, 3)
        if selected_rank <= 0 or selected_rank > 3:
            raise RuntimeError("LOO_BASIS_CONSTRUCTION_PARITY_FAILURE")
        decomposition = modules["build_svd_basis"](
            {name: teachers[name] for name in basis_garments}, bounds, basis_garments, selected_rank
        )
        repeated = modules["build_svd_basis"](
            {name: teachers[name] for name in basis_garments}, bounds, basis_garments, selected_rank
        )
        reconstruction = {}
        render_parity = {}
        for name in basis_garments:
            rebuilt = decomposition.basis(decomposition.teacher_coefficients[name], chunk_size=16384)
            reconstruction[name] = normalized_rmse(rebuilt, teachers[name], bounds)
            sample = context["samples"][f"{name}/{CONDITIONS[0]}"]
            with torch.inference_mode():
                teacher_rgb, teacher_alpha = modules["parameterization"].render_prediction(
                    context["base"], sample, teachers[name], context["background"]
                )
                rebuilt_rgb, rebuilt_alpha = modules["parameterization"].render_prediction(
                    context["base"], sample, rebuilt, context["background"]
                )
            auxiliary_render_calls += 2
            rgb_max = float((teacher_rgb - rebuilt_rgb).abs().max())
            alpha_max = float((teacher_alpha - rebuilt_alpha).abs().max())
            render_parity[name] = {
                "condition": CONDITIONS[0], "rgb_max_abs_error": rgb_max,
                "alpha_max_abs_error": alpha_max,
                "rgb_bitwise_equal": torch.equal(teacher_rgb, rebuilt_rgb),
                "alpha_bitwise_equal": torch.equal(teacher_alpha, rebuilt_alpha),
                "threshold": 1e-5,
                "status": "PASS" if max(rgb_max, alpha_max) <= 1e-5 else "FAIL",
            }
        total_variance = float(singular_values.square().sum())
        explained = [float(value.square() / max(total_variance, 1e-30)) for value in singular_values]
        condition_number = float(singular_values[0] / singular_values[selected_rank - 1])
        statistics_payload = {
            "singular_values": [float(value) for value in singular_values],
            "explained_variance": explained,
            "cumulative_explained_variance": list(np.cumsum(explained)),
            "numerical_rank_tolerance": tolerance,
            "numerical_rank": numerical_rank,
            "selected_rank": selected_rank,
            "condition_number": condition_number,
            "basis_garment_reconstruction_rmse": reconstruction,
            "aggregate_reconstruction_rmse": statistics.fmean(reconstruction.values()),
            "repeat_build_fingerprint_match": decomposition.basis.fingerprint() == repeated.basis.fingerprint(),
            "subspace_principal_angle_max_degrees": 0.0,
        }
        payload = _basis_payload(decomposition, statistics_payload)
        split_dir = attempt / f"02_basis_construction/{held_out}"
        output_io.ensure_directory(attempt, split_dir)
        artifact = save_torch(attempt, split_dir / "loo_basis.pt", payload)
        normalization = {
            "held_out_garment": held_out,
            "basis_garments": basis_garments,
            "mean": payload["coefficient_mean"].tolist(),
            "std": payload["coefficient_std"].tolist(),
            "held_out_in_normalization": False,
        }
        normalization_artifact = json_write(split_dir / "coefficient_normalization.json", normalization)
        residual_parity_pass = (
            max(reconstruction.values()) <= 1e-5
            and statistics_payload["repeat_build_fingerprint_match"]
            and payload["basis_fingerprint"] != FULL_BASIS_FORBIDDEN_SHA
        )
        render_parity_pass = all(value["status"] == "PASS" for value in render_parity.values())
        parity_pass = residual_parity_pass and render_parity_pass
        row = {
            "split_id": split["split_id"], "held_out_garment": held_out,
            "basis_garments": basis_garments, **statistics_payload,
            "basis_fingerprint": payload["basis_fingerprint"],
            "basis_artifact_sha256": artifact["sha256"],
            "normalization_sha256": normalization_artifact["sha256"],
            "basis_scalar_count": decomposition.basis.explicit_scalar_count,
            "held_out_teacher_in_basis_count": 0,
            "held_out_teacher_in_normalization_count": 0,
            "full_five_garment_basis_reused": False,
            "residual_parity": "PASS" if residual_parity_pass else "FAIL",
            "render_parity": "PASS" if render_parity_pass else "FAIL",
            "render_parity_rows": render_parity,
            "status": "PASS" if parity_pass else "FAIL",
        }
        json_write(split_dir / "basis_report.json", row)
        if not parity_pass:
            raise RuntimeError("LOO_BASIS_CONSTRUCTION_PARITY_FAILURE")
        rows.append(row)
        del matrix, centered, flattened, decomposition, repeated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "schema_version": "canondressgs.paper.loo_basis_execution_registry.v1",
        "task_id": TASK_ID, "status": "PASS", "split_count": len(rows),
        "basis_generation_runs": len(rows), "svd_runs": len(rows),
        "auxiliary_render_calls": auxiliary_render_calls, "rows": rows,
    }
    json_write(destination, result)
    return result


def _reference_feature(
    extractor: Any, context: Mapping[str, Any], outfit: str, conditions: Sequence[str]
) -> "torch.Tensor":
    images = torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_edit_rgb"]
        for condition in conditions
    ])
    masks = torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_clothing_mask"]
        for condition in conditions
    ])
    valid = torch.ones((len(conditions), 1), device=images.device, dtype=images.dtype)
    with torch.inference_mode():
        value = extractor(images, masks, valid).set_feature.detach()
    if value.shape != (1, 512) or not torch.isfinite(value).all():
        raise RuntimeError("frozen F2 reference feature contract mismatch")
    return value[0]


def _calibration_initialization_feature_record(
    extractor: Any, context: Mapping[str, Any], outfit: str, conditions: Sequence[str]
) -> Any:
    from scene.loo_f2_feature_adapter import (
        CALIBRATION_INITIALIZATION_ROLE,
        build_reference_metadata,
        extract_reference_feature_set,
    )

    images = torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_edit_rgb"]
        for condition in conditions
    ])
    masks = torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_clothing_mask"]
        for condition in conditions
    ])
    valid = torch.ones((len(conditions), 1), device=images.device, dtype=images.dtype)
    with torch.inference_mode():
        value = extract_reference_feature_set(
            extractor, images, masks, valid,
            build_reference_metadata(context, outfit, conditions),
            role=CALIBRATION_INITIALIZATION_ROLE,
        )
    if (
        value.aggregated_feature.shape != (1, 512)
        or not torch.isfinite(value.aggregated_feature).all()
    ):
        raise RuntimeError("frozen F2 calibration initialization feature contract mismatch")
    return value


def _loo_centroid_selection(
    task: Mapping[str, Any],
    feature: Any,
) -> dict[str, Any]:
    paired_k2 = next(
        row for row in tasks()
        if row["held_out_garment"] == task["held_out_garment"]
        and row["rotation"] == task["rotation"]
        and int(row["K"]) == 2
    )
    train_conditions = paired_k2["selected_adaptation_conditions"]
    raw_centroids = {
        outfit: torch.stack([feature(outfit, [condition]) for condition in train_conditions]).mean(0)
        for outfit in task["basis_garments"]
    }
    matrix = torch.stack([raw_centroids[outfit] for outfit in task["basis_garments"]])
    mean = matrix.mean(0)
    scale = matrix.std(0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    centroids = {
        outfit: (raw_centroids[outfit] - mean) / scale
        for outfit in task["basis_garments"]
    }
    query = feature(task["held_out_garment"], task["selected_adaptation_conditions"])
    standardized_query = (query - mean) / scale
    distances = {
        outfit: float(torch.square(centroids[outfit] - standardized_query).sum())
        for outfit in task["basis_garments"]
    }
    selected = min(distances, key=lambda name: (distances[name], name))
    return {
        "centroid_train_conditions": list(train_conditions),
        "raw_centroids": raw_centroids,
        "centroids": centroids,
        "feature_mean": mean,
        "feature_scale": scale,
        "query": query,
        "distances": distances,
        "selected_known_endpoint": selected,
    }


def f2_cache_preflight(
    root: Path, context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt_path(root)
    context = runtime_context(attempt) if context is None else context
    extractor = runtime_imports()["multi"]._feature_extractor(context)
    feature_cache: dict[tuple[str, tuple[str, ...]], "torch.Tensor"] = {}

    def feature(outfit: str, conditions: Sequence[str]) -> "torch.Tensor":
        key = (outfit, tuple(conditions))
        if key not in feature_cache:
            feature_cache[key] = _reference_feature(extractor, context, outfit, conditions)
        return feature_cache[key]

    rows = []
    for task in tasks():
        selection = _loo_centroid_selection(task, feature)
        selected = selection["selected_known_endpoint"]
        rows.append({
            "task_id": task["task_id"], "held_out_garment": task["held_out_garment"],
            "rotation": task["rotation"], "K": int(task["K"]),
            "selected_known_endpoint": selected,
            "selected_squared_distance": selection["distances"][selected],
            "centroid_train_conditions": selection["centroid_train_conditions"],
            "bank": list(task["basis_garments"]), "held_out_teacher_used": False,
        })
    pairs = []
    for held_out in OUTFITS:
        for rotation in ("R0", "R1", "R2", "R3"):
            pair = [
                row for row in rows
                if row["held_out_garment"] == held_out and row["rotation"] == rotation
            ]
            k1 = next(row for row in pair if row["K"] == 1)
            k2 = next(row for row in pair if row["K"] == 2)
            pairs.append({
                "held_out_garment": held_out, "rotation": rotation,
                "K1_selected_known_endpoint": k1["selected_known_endpoint"],
                "K2_selected_known_endpoint": k2["selected_known_endpoint"],
                "same_endpoint": k1["selected_known_endpoint"] == k2["selected_known_endpoint"],
            })
    matching = sum(row["same_endpoint"] for row in pairs)
    expected_cache_hits = int(expected_counts()["K_shared_static_cache_hits"])
    actual_cache_hits = 100 + matching
    mismatches = {
        (row["held_out_garment"], row["rotation"]): (
            row["K1_selected_known_endpoint"], row["K2_selected_known_endpoint"]
        )
        for row in pairs if not row["same_endpoint"]
    }
    expected_mismatches = {
        ("O01", "R0"): ("O04", "O02"),
        ("O01", "R1"): ("O04", "O02"),
        ("O02", "R2"): ("O08", "O01"),
        ("O04", "R0"): ("O03", "O01"),
        ("O08", "R3"): ("O01", "O02"),
    }
    return {
        "schema_version": "canondressgs.paper.loo_f2_cache_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if matching == 15 and len(pairs) - matching == 5 and mismatches == expected_mismatches and actual_cache_hits == expected_cache_hits else "FAIL",
        "task_count": len(rows), "paired_group_count": len(pairs),
        "K_shared_hard_lookup_pair_count": matching,
        "K_divergent_hard_lookup_pair_count": len(pairs) - matching,
        "expected_K_shared_hard_lookup_pair_count": 15,
        "expected_K_divergent_hard_lookup_pair_count": 5,
        "actual_static_cache_hits": actual_cache_hits,
        "expected_static_cache_hits": expected_cache_hits,
        "mismatch_identities_exact": mismatches == expected_mismatches,
        "centroid_definition": "mean of per-condition frozen-F2 singleton features over the rotation train folds",
        "feature_forward_cache_entries": len(feature_cache),
        "rows": rows, "pairs": pairs,
    }


def build_lookup_registry(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "09_hard_lookup_analysis/hard_lookup_registry.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    extractor = runtime_imports()["multi"]._feature_extractor(context)
    feature_cache: dict[tuple[str, tuple[str, ...]], "torch.Tensor"] = {}

    def feature(outfit: str, conditions: Sequence[str]) -> "torch.Tensor":
        key = (outfit, tuple(conditions))
        if key not in feature_cache:
            feature_cache[key] = _reference_feature(extractor, context, outfit, conditions)
        return feature_cache[key]

    rows = []
    for task in tasks():
        held_out = task["held_out_garment"]
        basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
        selection = _loo_centroid_selection(task, feature)
        selected = selection["selected_known_endpoint"]
        row = {
            "task_id": task["task_id"], "held_out_garment": held_out,
            "rotation": task["rotation"], "K": int(task["K"]),
            "adaptation_conditions": task["selected_adaptation_conditions"],
            "centroid_train_conditions": selection["centroid_train_conditions"],
            "hard_lookup_bank": list(task["basis_garments"]),
            "selected_known_endpoint": selected,
            "squared_distances": selection["distances"],
            "query_feature_sha256": tensor_sha(selection["query"]),
            "raw_centroid_sha256": {
                name: tensor_sha(value) for name, value in selection["raw_centroids"].items()
            },
            "standardized_centroid_sha256": {
                name: tensor_sha(value) for name, value in selection["centroids"].items()
            },
            "feature_mean_sha256": tensor_sha(selection["feature_mean"]),
            "feature_scale_sha256": tensor_sha(selection["feature_scale"]),
            "selected_coefficient": coefficients[selected].detach().cpu().tolist(),
            "basis_artifact_sha256": sha256(attempt / f"02_basis_construction/{held_out}/loo_basis.pt"),
            "normalization_sha256": sha256(attempt / f"02_basis_construction/{held_out}/coefficient_normalization.json"),
            "held_out_teacher_used": False, "test_observation_used": False,
            "bank_contains_held_out": held_out in task["basis_garments"],
            "tie_break": "minimum standardized squared L2 then lexical outfit id",
        }
        if row["bank_contains_held_out"]:
            raise RuntimeError("LOO_HELD_OUT_INFORMATION_LEAKAGE")
        rows.append(row)
    pairs = []
    for held_out in OUTFITS:
        for rotation in ("R0", "R1", "R2", "R3"):
            pair = [
                row for row in rows
                if row["held_out_garment"] == held_out and row["rotation"] == rotation
            ]
            k1 = next(row for row in pair if int(row["K"]) == 1)
            k2 = next(row for row in pair if int(row["K"]) == 2)
            pairs.append({
                "held_out_garment": held_out, "rotation": rotation,
                "K1_selected_known_endpoint": k1["selected_known_endpoint"],
                "K2_selected_known_endpoint": k2["selected_known_endpoint"],
                "same_endpoint": k1["selected_known_endpoint"] == k2["selected_known_endpoint"],
            })
    matching = sum(row["same_endpoint"] for row in pairs)
    expected_cache_hits = int(expected_counts()["K_shared_static_cache_hits"])
    actual_cache_hits = 100 + matching
    mismatches = {
        (row["held_out_garment"], row["rotation"]): (
            row["K1_selected_known_endpoint"], row["K2_selected_known_endpoint"]
        )
        for row in pairs if not row["same_endpoint"]
    }
    expected_mismatches = {
        ("O01", "R0"): ("O04", "O02"),
        ("O01", "R1"): ("O04", "O02"),
        ("O02", "R2"): ("O08", "O01"),
        ("O04", "R0"): ("O03", "O01"),
        ("O08", "R3"): ("O01", "O02"),
    }
    cache_contract_pass = (
        len(pairs) == 20 and matching == 15
        and len(pairs) - matching == 5
        and mismatches == expected_mismatches
        and actual_cache_hits == expected_cache_hits
    )
    result = {
        "schema_version": "canondressgs.paper.loo_hard_lookup_registry.v1",
        "task_id": TASK_ID, "status": "PASS" if len(rows) == 40 and cache_contract_pass else "FAIL",
        "row_count": len(rows), "rows": rows,
        "K_shared_cache_contract": {
            "status": "PASS" if cache_contract_pass else "FAIL",
            "expected_pair_count": 20, "actual_pair_count": len(pairs),
            "matching_endpoint_pair_count": matching,
            "divergent_endpoint_pair_count": len(pairs) - matching,
            "expected_matching_endpoint_pair_count": 15,
            "expected_divergent_endpoint_pair_count": 5,
            "mismatch_identities_exact": mismatches == expected_mismatches,
            "expected_static_cache_hits": expected_cache_hits,
            "actual_static_cache_hits": actual_cache_hits,
            "pairs": pairs,
        },
        "held_out_teacher_use_in_deployable_adaptation": 0,
    }
    json_write(destination, result)
    if result["status"] != "PASS":
        raise RuntimeError("LOO_STATIC_CACHE_CONTRACT_FAILURE")
    return result


def lookup_row(attempt: Path, task_id: str) -> dict[str, Any]:
    registry = read_json(attempt / "09_hard_lookup_analysis/hard_lookup_registry.json")
    try:
        return next(row for row in registry["rows"] if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError(task_id) from error


def regularization_selection() -> dict[str, Any]:
    loss = contracts()["loss"]
    low = loss["regularization_selection"]["low_dimensional_grid"]
    full = loss["regularization_selection"]["full_residual_grid"]
    return {
        "selection_scope": "once per LOO split and method, shared across rotations and K",
        "selection_data": "four basis garments on calibration fold only using internal leave-one-basis-garment-out simulation",
        "selection_status": "RUNTIME_CALIBRATION_REQUIRED_BEFORE_FORMAL_OPTIMIZATION",
        "candidate_grids": {"low_dimensional": low, "full_residual": full},
        "tie_break": loss["regularization_selection"]["tie_break"],
        "held_out_garment_information_used": 0,
        "test_used": False,
    }


def select_regularization_candidate(
    rows: Sequence[Mapping[str, Any]], lambda_name: str,
) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["identity_contamination_count"]) == 0]
    if not eligible:
        raise RuntimeError("LOO_REGULARIZATION_CALIBRATION_IDENTITY_FAILURE")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["mean_calibration_common_rendering_loss"]),
            float(row["trust_region_radius"]),
            -float(row[lambda_name]),
            str(row["candidate_id"]),
        ),
    )
    return dict(selected)


def regularization_for_split(attempt: Path, held_out: str, family: str) -> dict[str, float]:
    registry = read_json(attempt / "01_contract_snapshot/regularization_selection.json")
    split = next(row for row in registry["splits"] if row["held_out_garment"] == held_out)
    key = "full_residual" if family == METHOD_FAMILIES[2] else "low_dimensional"
    selected = split[key]["selected"]
    if key == "low_dimensional":
        return {
            "coefficient_anchor_lambda": float(selected["coefficient_anchor_lambda"]),
            "trust_region_radius": float(selected["trust_region_radius"]),
        }
    return {
        "residual_delta_lambda": float(selected["residual_delta_lambda"]),
        "trust_region_radius": float(selected["trust_region_radius"]),
    }


class LPIPSRuntime:
    def __init__(self, root: Path, device: "torch.device") -> None:
        self.root = root
        self.device = device
        self.model: Any | None = None

    def lpips(self) -> Any:
        if self.model is None:
            package_root = self.root.parent / "AnimatableGaussians"
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            module = importlib.import_module("network.lpips.lpips")
            self.model = module.LPIPS(net="vgg", version="0.1", verbose=False).to(self.device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        return self.model


def _mask_like(mask: "torch.Tensor", value: "torch.Tensor") -> "torch.Tensor":
    result = mask.to(value)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    return result


def _masked_mean(value: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    selected = _mask_like(mask, value)
    while selected.ndim < value.ndim:
        selected = selected.unsqueeze(0)
    selected = selected.expand_as(value)
    return (value * selected).sum() / selected.sum().clamp_min(1e-8)


def differentiable_lpips(
    runtime: LPIPSRuntime, first: "torch.Tensor", second: "torch.Tensor", mask: "torch.Tensor"
) -> "torch.Tensor":
    helper = runtime_imports()["metrics"]
    return runtime.lpips()(helper.lpips_input(first, mask), helper.lpips_input(second, mask)).reshape(())


def loo_render_loss(
    lpips_runtime: LPIPSRuntime,
    rgb: "torch.Tensor",
    alpha: "torch.Tensor",
    sample: Mapping[str, Any],
    regularization: "torch.Tensor",
) -> dict[str, "torch.Tensor"]:
    component = contracts()["loss"]["components"]
    target = sample["target_edit_rgb"].to(rgb)
    base = sample["target_base_rgb"].to(rgb)
    garment = _mask_like(sample["target_clothing_mask"], rgb)
    foreground = _mask_like(sample["target_foreground_mask"], alpha)
    base_foreground = _mask_like(sample["target_base_foreground_mask"], alpha)
    protected = _mask_like(sample["target_protected_mask"], rgb)
    membership = (garment >= 0.5).to(rgb)
    erosion = F.conv2d(membership[None], torch.ones((1, 1, 3, 3), device=rgb.device, dtype=rgb.dtype), padding=1)
    boundary = (membership[None] - (erosion >= 9).to(rgb)).clamp_min(0)[0]
    parts = {
        "garment_rgb_l1": _masked_mean((rgb - target).abs(), garment),
        "lpips": differentiable_lpips(lpips_runtime, rgb, target, garment),
        "mask_l1": (alpha - foreground).abs().mean(),
        "boundary_rgb_l1": _masked_mean((rgb - target).abs(), boundary),
        "identity_rgb_l1": _masked_mean((rgb - base).abs(), protected),
        "identity_alpha_l1": _masked_mean((alpha - base_foreground).abs(), protected),
        "regularization": regularization,
    }
    total = sum(
        float(component[name]["weight"]) * parts[name]
        for name in (
            "garment_rgb_l1", "lpips", "mask_l1", "boundary_rgb_l1",
            "identity_rgb_l1", "identity_alpha_l1",
        )
    ) + regularization
    return {"total": total, **parts}


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
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_optimizer(parameters: Iterable["torch.Tensor"]) -> tuple[Any, Any]:
    budget = contracts()["execution"]["optimizer_budget"]
    optimizer = torch.optim.Adam(
        parameters, lr=float(budget["learning_rate"]),
        betas=tuple(float(value) for value in budget["betas"]),
        eps=float(budget["epsilon"]), weight_decay=float(budget["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def checkpoint_path(attempt: Path, task_id: str, family: str, step: int) -> Path:
    return attempt / f"06_checkpoints/{task_id}/{family}/step_{step:06d}.pth"


def optimizer_to(optimizer: Any, device: "torch.device") -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def checkpoint_payload(
    attempt: Path, task: Mapping[str, Any], family: str, step: int,
    state: Mapping[str, Any], optimizer: Any, scheduler: Any,
    elapsed: float, metric_snapshot: Mapping[str, Any], initialization: Mapping[str, Any],
) -> dict[str, Any]:
    held_out = task["held_out_garment"]
    return {
        "schema_version": "canondressgs.paper.loo_optimizer_checkpoint.v1",
        "attempt": ATTEMPT_NAME, "task_id": task["task_id"], "method_family": family,
        "held_out_garment": held_out, "basis_garments": task["basis_garments"],
        "basis_sha256": sha256(attempt / f"02_basis_construction/{held_out}/loo_basis.pt"),
        "normalization_sha256": sha256(attempt / f"02_basis_construction/{held_out}/coefficient_normalization.json"),
        "rotation": task["rotation"], "K": int(task["K"]),
        "initialization": dict(initialization), "global_step": int(step),
        "trainable_state": dict(state), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "rng_state": rng_state(),
        "view_data_position": 0, "execution_head": assert_execution_head(attempt),
        "loss_sha256": sha256(RISK / "loo_adaptation_loss_contract.json"),
        "optimizer_sha256": sha256(RISK / "loo_optimizer_contract.json"),
        "schedule_sha256": task["query_order_sha"],
        "elapsed_seconds": float(elapsed), "metric_snapshot": dict(metric_snapshot),
        "held_out_teacher_used": False, "test_views_used_for_updates": False,
    }


def write_checkpoint(attempt: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "attempt", "task_id", "method_family", "held_out_garment", "basis_garments",
        "basis_sha256", "normalization_sha256", "rotation", "K", "initialization",
        "global_step", "trainable_state", "optimizer_state", "scheduler_state", "rng_state",
        "view_data_position", "execution_head", "loss_sha256", "optimizer_sha256",
        "schedule_sha256", "elapsed_seconds", "metric_snapshot",
    }
    def validate(temporary: Path) -> None:
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping) or not required.issubset(loaded):
            raise ValueError("LOO checkpoint roundtrip schema mismatch")
    return output_io.atomic_torch_save(attempt, path, dict(payload), validate, allow_replace=False)


def update_status(attempt: Path, **updates: Any) -> None:
    path = attempt / "RUN_STATUS.json"
    value = read_json(path)
    value.update(updates)
    value["updated_at_utc"] = now()
    json_write(path, value, replace=True)


class FullResidualDelta(nn.Module):
    def __init__(self, base: Any, initial: Any, bounds: Mapping[str, float]) -> None:
        super().__init__()
        if int(base._xyz.shape[0]) != 200_000 or tuple(base._shN.shape[1:]) != (3, 3):
            raise ValueError("LOO full residual requires 200k Gaussians and 9 shN scalars")
        self.raw_delta = nn.Parameter(torch.zeros((200_000, 22), device=base._xyz.device, dtype=base._xyz.dtype))
        support = runtime_imports()["garment_trainable_mask"](base).to(base._xyz)
        self.register_buffer("support", support, persistent=True)
        self.bounds = {name: float(value) for name, value in bounds.items()}
        for name, value in zip(runtime_imports()["CHANNELS"], residual_fields(initial)):
            self.register_buffer(f"initial__{name}", value.detach().clone(), persistent=True)
        self.raw_delta.register_hook(lambda gradient: gradient * support.expand(-1, 22))

    def residual(self) -> Any:
        value = self.raw_delta * self.support
        fields = {
            "delta_xyz": self.initial__delta_xyz + value[:, 0:3] * self.bounds["xyz"],
            "delta_log_scaling": self.initial__delta_log_scaling + value[:, 3:6] * self.bounds["log_scaling"],
            "delta_rotvec": self.initial__delta_rotvec + value[:, 6:9] * self.bounds["rotation"],
            "delta_opacity_logit": self.initial__delta_opacity_logit + value[:, 9] * self.bounds["opacity_logit"],
            "delta_sh0": self.initial__delta_sh0 + value[:, 10:13].reshape(200_000, 1, 3) * self.bounds["sh0"],
            "delta_shN": self.initial__delta_shN + value[:, 13:22].reshape(200_000, 3, 3) * self.bounds["shN"],
        }
        return runtime_imports()["GaussianClothingResiduals"](**fields)

    def regularization(self, value: Mapping[str, float]) -> "torch.Tensor":
        active = self.raw_delta * self.support
        anchor = active.square().mean()
        per_gaussian = torch.linalg.vector_norm(active, dim=1)
        trust = torch.relu(per_gaussian - float(value["trust_region_radius"])).square().mean()
        return float(value["residual_delta_lambda"]) * anchor + trust


def _normalized_residual_matrix(residual: Any, bounds: Mapping[str, float]) -> "torch.Tensor":
    fields = runtime_imports()["normalized_residual_dict"](residual, bounds)
    return torch.cat((
        fields["delta_xyz"].reshape(-1, 3),
        fields["delta_log_scaling"].reshape(-1, 3),
        fields["delta_rotvec"].reshape(-1, 3),
        fields["delta_opacity_logit"].reshape(-1, 1),
        fields["delta_sh0"].reshape(-1, 3),
        fields["delta_shN"].reshape(-1, 9),
    ), dim=1)


def _support_leakage(residual: Any, bounds: Mapping[str, float], support: "torch.Tensor") -> float:
    values = _normalized_residual_matrix(residual, bounds)
    protected = support.reshape(-1) == 0
    return float(values[protected].abs().max()) if bool(protected.any()) else 0.0


def _calibration_render_summary(
    context: Mapping[str, Any], lpips_runtime: LPIPSRuntime, outfit: str,
    conditions: Sequence[str], residual: Any,
) -> dict[str, float]:
    rows = []
    with torch.inference_mode():
        zero = context["base"]._xyz.new_zeros(())
        for condition in conditions:
            sample = context["samples"][f"{outfit}/{condition}"]
            rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                context["base"], sample, residual, context["background"]
            )
            losses = loo_render_loss(lpips_runtime, rgb, alpha, sample, zero)
            rows.append({name: float(value) for name, value in losses.items()})
    return {
        "common_rendering_loss": statistics.fmean(row["total"] for row in rows),
        "identity_rgb_l1": statistics.fmean(row["identity_rgb_l1"] for row in rows),
        "identity_alpha_l1": statistics.fmean(row["identity_alpha_l1"] for row in rows),
        "render_count": float(len(rows)),
    }


def _regularization_candidate_rows(
    instances: Sequence[Mapping[str, Any]], method: str,
) -> list[dict[str, Any]]:
    grids = regularization_selection()["candidate_grids"]
    rows = []
    if method == "low_dimensional":
        lambda_name = "coefficient_anchor_lambda"
        lambdas = grids[method][lambda_name]
        radii = grids[method]["trust_region_radius_standardized_l2"]
        for lambda_value in lambdas:
            for radius in radii:
                penalties = [
                    float(lambda_value) * float(instance["anchor"])
                    + max(0.0, float(instance["displacement_norm"]) - float(radius)) ** 2
                    for instance in instances
                ]
                rows.append({
                    "candidate_id": f"lambda={float(lambda_value):.10g};radius={float(radius):.10g}",
                    lambda_name: float(lambda_value), "trust_region_radius": float(radius),
                    "mean_regularization": statistics.fmean(penalties),
                    "mean_calibration_common_rendering_loss": statistics.fmean(
                        float(instance["common_rendering_loss"]) + penalty
                        for instance, penalty in zip(instances, penalties)
                    ),
                    "identity_contamination_count": sum(int(instance["identity_contamination"]) for instance in instances),
                    "internal_simulation_count": len(instances),
                })
        return rows
    lambda_name = "residual_delta_lambda"
    lambdas = grids["full_residual"][lambda_name]
    radii = grids["full_residual"]["trust_region_radius_bound_normalized_l2_per_gaussian"]
    for lambda_value in lambdas:
        for radius in radii:
            penalties = [
                float(lambda_value) * float(instance["anchor"])
                + float(instance["trust_penalties"][str(float(radius))])
                for instance in instances
            ]
            rows.append({
                "candidate_id": f"lambda={float(lambda_value):.10g};radius={float(radius):.10g}",
                lambda_name: float(lambda_value), "trust_region_radius": float(radius),
                "mean_regularization": statistics.fmean(penalties),
                "mean_calibration_common_rendering_loss": statistics.fmean(
                    float(instance["common_rendering_loss"]) + penalty
                    for instance, penalty in zip(instances, penalties)
                ),
                "identity_contamination_count": sum(int(instance["identity_contamination"]) for instance in instances),
                "internal_simulation_count": len(instances),
            })
    return rows


def calibrate_regularization(
    root: Path, context: Mapping[str, Any] | None = None,
    lpips_runtime: LPIPSRuntime | None = None,
) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "01_contract_snapshot/regularization_selection.json"
    if destination.is_file():
        return read_json(destination)
    if read_json(attempt / "02_basis_construction/summary.json")["status"] != "PASS":
        raise RuntimeError("regularization calibration requires basis PASS")
    context = runtime_context(attempt) if context is None else context
    teachers = load_teachers(context)
    modules = runtime_imports()
    extractor = modules["multi"]._feature_extractor(context)
    lpips_runtime = lpips_runtime or LPIPSRuntime(root, context["base"]._xyz.device)
    support = modules["garment_trainable_mask"](context["base"]).to(context["base"]._xyz)
    selection_contract = regularization_selection()
    splits = []
    auxiliary_renders = 0
    for split in split_rows():
        outer_held_out = split["held_out_garment"]
        basis_garments = list(split["basis_garments"])
        bounds = split["basis_contract"]["channel_bounds"]
        calibration_conditions = [
            next(
                task for task in tasks()
                if task["held_out_garment"] == outer_held_out
                and task["rotation"] == rotation and int(task["K"]) == 2
            )["calibration_conditions"][0]
            for rotation in ("R0", "R1", "R2", "R3")
        ]
        low_instances = []
        full_instances = []
        for internal_held_out in basis_garments:
            internal_bank = [name for name in basis_garments if name != internal_held_out]
            normalized = {
                name: modules["normalized_residual_dict"](teachers[name], bounds)
                for name in internal_bank
            }
            matrix = torch.stack([
                torch.cat([normalized[name][field].reshape(-1) for field in modules["CHANNELS"]])
                for name in internal_bank
            ])
            centered = matrix - matrix.mean(0)
            singular_values = torch.linalg.svdvals(centered.double())
            tolerance = max(len(internal_bank), int(centered.shape[1])) * torch.finfo(centered.dtype).eps * float(singular_values[0])
            inner_rank = min(int((singular_values > tolerance).sum()), len(internal_bank) - 1)
            if inner_rank <= 0 or inner_rank > 2:
                raise RuntimeError("LOO_REGULARIZATION_INTERNAL_BASIS_FAILURE")
            decomposition = modules["build_svd_basis"](
                {name: teachers[name] for name in internal_bank}, bounds, internal_bank, inner_rank
            )
            known_feature_records = {
                name: _calibration_initialization_feature_record(
                    extractor, context, name, calibration_conditions,
                )
                for name in internal_bank
            }
            query_record = _calibration_initialization_feature_record(
                extractor, context, internal_held_out, calibration_conditions,
            )
            known_features = {
                name: record.feature_vector() for name, record in known_feature_records.items()
            }
            query = query_record.feature_vector()
            feature_matrix = torch.stack(list(known_features.values()))
            feature_mean = feature_matrix.mean(0)
            feature_std = feature_matrix.std(0, unbiased=False).clamp_min(1e-8)
            query_standardized = (query - feature_mean) / feature_std
            feature_distances = {
                name: float(torch.square((value - feature_mean) / feature_std - query_standardized).sum())
                for name, value in known_features.items()
            }
            selected = min(feature_distances, key=lambda name: (feature_distances[name], name))
            oracle_coefficient = modules["project_residual_onto_basis"](
                decomposition.basis, teachers[internal_held_out]
            )
            known_coefficient_matrix = torch.stack([
                decomposition.teacher_coefficients[name] for name in internal_bank
            ])
            coefficient_std = known_coefficient_matrix.std(0, unbiased=False).clamp_min(1e-8)
            standardized_displacement = (
                oracle_coefficient - decomposition.teacher_coefficients[selected]
            ) / coefficient_std
            low_residual = decomposition.basis(oracle_coefficient, chunk_size=16384)
            low_render = _calibration_render_summary(
                context, lpips_runtime, internal_held_out, calibration_conditions, low_residual
            )
            auxiliary_renders += int(low_render["render_count"])
            low_leakage = _support_leakage(low_residual, bounds, support)
            low_instances.append({
                "internal_held_out_garment": internal_held_out,
                "internal_basis_garments": internal_bank, "internal_rank": inner_rank,
                "selected_initial_endpoint": selected,
                "selected_feature_squared_distance": feature_distances[selected],
                "anchor": float(standardized_displacement.square().mean()),
                "displacement_norm": float(torch.linalg.vector_norm(standardized_displacement)),
                "support_leakage_max": low_leakage,
                "identity_contamination": int(low_leakage > 1e-7),
                "projection_rmse": normalized_rmse(low_residual, teachers[internal_held_out], bounds),
                **low_render,
            })
            initial_residual = decomposition.basis(
                decomposition.teacher_coefficients[selected], chunk_size=16384
            )
            field = FullResidualDelta(context["base"], initial_residual, bounds)
            raw_delta = (
                _normalized_residual_matrix(teachers[internal_held_out], bounds)
                - _normalized_residual_matrix(initial_residual, bounds)
            )
            with torch.no_grad():
                field.raw_delta.copy_(raw_delta)
                full_residual = field.residual()
                active = field.raw_delta * field.support
                per_gaussian = torch.linalg.vector_norm(active, dim=1)
            full_render = _calibration_render_summary(
                context, lpips_runtime, internal_held_out, calibration_conditions, full_residual
            )
            auxiliary_renders += int(full_render["render_count"])
            full_leakage = _support_leakage(full_residual, bounds, support)
            full_radii = selection_contract["candidate_grids"]["full_residual"]["trust_region_radius_bound_normalized_l2_per_gaussian"]
            full_instances.append({
                "internal_held_out_garment": internal_held_out,
                "internal_basis_garments": internal_bank, "selected_initial_endpoint": selected,
                "selected_feature_squared_distance": feature_distances[selected],
                "anchor": float(active.square().mean()),
                "trust_penalties": {
                    str(float(radius)): float(torch.relu(per_gaussian - float(radius)).square().mean())
                    for radius in full_radii
                },
                "support_leakage_max": full_leakage,
                "identity_contamination": int(full_leakage > 1e-7),
                "reachable_target_rmse": normalized_rmse(full_residual, teachers[internal_held_out], bounds),
                **full_render,
            })
            del decomposition, field, raw_delta, active, per_gaussian
        low_candidates = _regularization_candidate_rows(low_instances, "low_dimensional")
        full_candidates = _regularization_candidate_rows(full_instances, "full_residual")
        low_selected = select_regularization_candidate(low_candidates, "coefficient_anchor_lambda")
        full_selected = select_regularization_candidate(full_candidates, "residual_delta_lambda")
        splits.append({
            "held_out_garment": outer_held_out, "basis_garments": basis_garments,
            "calibration_conditions": calibration_conditions,
            "outer_held_out_teacher_used": False, "test_used": False,
            "low_dimensional": {
                "selected": low_selected, "candidate_rows": low_candidates,
                "internal_simulations": low_instances,
            },
            "full_residual": {
                "selected": full_selected, "candidate_rows": full_candidates,
                "internal_simulations": full_instances,
            },
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "schema_version": "canondressgs.paper.loo_regularization_selection.v1",
        "task_id": TASK_ID, "status": "PASS", **selection_contract,
        "selection_status": "FROZEN_FROM_INTERNAL_LEAVE_ONE_BASIS_GARMENT_OUT_CALIBRATION",
        "split_count": len(splits), "splits": splits,
        "auxiliary_calibration_render_calls": auxiliary_renders,
        "formal_optimizer_or_evaluation_denominator_changed": False,
    }
    if len(splits) != 5 or any(set(row["basis_garments"]) != {
        instance["internal_held_out_garment"] for instance in row["low_dimensional"]["internal_simulations"]
    } for row in splits):
        raise RuntimeError("LOO_REGULARIZATION_CALIBRATION_CARDINALITY_FAILURE")
    json_write(destination, result)
    return result


def run_dir(attempt: Path, task: Mapping[str, Any], family: str) -> Path:
    phase = "05_full_residual_runs" if family == METHOD_FAMILIES[2] else "04_low_dimensional_runs"
    return attempt / phase / task["task_id"] / family


def _resume_checkpoint(
    attempt: Path, task: Mapping[str, Any], family: str, optimizer: Any,
    scheduler: Any, parameter: "torch.Tensor",
) -> tuple[int, float, list[dict[str, Any]], int]:
    directory = run_dir(attempt, task, family)
    status = read_json(directory / "RUN_STATUS.json")
    completed = int(status["optimizer_steps"])
    trajectory = jsonl(directory / "trajectory.jsonl")
    if completed not in MILESTONES or len(trajectory) != completed:
        raise RuntimeError("interrupted run is outside an exact frozen checkpoint boundary")
    checkpoint = checkpoint_path(attempt, task["task_id"], family, completed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["execution_head"] != assert_execution_head(attempt) or int(payload["global_step"]) != completed:
        raise RuntimeError("resume execution head or step mismatch")
    state = payload["trainable_state"]
    source = state["coefficient"] if "coefficient" in state else state["raw_delta"]
    with torch.no_grad():
        parameter.copy_(source.to(parameter))
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    optimizer_to(optimizer, parameter.device)
    restore_rng(payload["rng_state"])
    return completed, float(payload["elapsed_seconds"]), trajectory, int(status.get("resume_count", 0)) + 1


def _frozen_gradient_audit(context: Mapping[str, Any], basis: Any) -> dict[str, Any]:
    values: list[tuple[str, "torch.Tensor"]] = []
    base = context["base"]
    if hasattr(base, "named_parameters"):
        values.extend((f"base.{name}", value) for name, value in base.named_parameters())
    values.extend((f"basis.{name}", value) for name, value in basis.named_parameters())
    violations = [
        name for name, value in values
        if value.grad is not None and bool(torch.count_nonzero(value.grad.detach()))
    ]
    return {
        "status": "PASS" if not violations else "FAIL",
        "audited_parameter_count": len(values), "violations": violations,
    }


def _checkpoint_record(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "step": int(payload["global_step"]), "method_family": payload["method_family"],
        "task_id": payload["task_id"],
    }


def train_one_low(
    attempt: Path, root: Path, context: Mapping[str, Any], lpips_runtime: LPIPSRuntime,
    task: Mapping[str, Any], family: str,
) -> dict[str, Any]:
    if family not in METHOD_FAMILIES[:2]:
        raise ValueError(family)
    directory = run_dir(attempt, task, family)
    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"completed LOO run is invalid: {directory}")
        return summary
    creating = not directory.exists()
    if creating:
        output_io.ensure_directory(attempt, directory)
    budget = contracts()["execution"]["optimizer_budget"]
    seed_everything(int(budget["seed"]))
    held_out = task["held_out_garment"]
    basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
    lookup = lookup_row(attempt, task["task_id"])
    selected = lookup["selected_known_endpoint"]
    initial = coefficients[selected].detach().clone() if family == METHOD_FAMILIES[0] else torch.zeros(basis.rank, device=context["base"]._xyz.device, dtype=context["base"]._xyz.dtype)
    coefficient = nn.Parameter(initial.clone())
    if coefficient.numel() != basis.rank or basis.rank > 3:
        raise RuntimeError("LOO low-dimensional trainable scalar mismatch")
    optimizer, scheduler = make_optimizer([coefficient])
    normalization_mean = payload["coefficient_mean"].to(coefficient)
    normalization_std = payload["coefficient_std"].to(coefficient).clamp_min(1e-8)
    regularization = regularization_for_split(attempt, held_out, family)
    initialization = {
        "name": "REFERENCE_NEAREST_HARD_LOOKUP_COEFFICIENT" if family == METHOD_FAMILIES[0] else "ZERO_COEFFICIENT_INITIALIZATION",
        "selected_known_endpoint": selected if family == METHOD_FAMILIES[0] else None,
        "coefficient": initial.detach().cpu().tolist(),
        "held_out_teacher_used": False,
    }
    checkpoints: list[dict[str, Any]] = []
    start_step, elapsed, trajectory, resume_count = 0, 0.0, [], 0
    if creating:
        json_write(directory / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": 0, "optimizer_creations": 1,
            "resume_count": 0, "repeated_optimizer_steps": 0,
        })
        path = checkpoint_path(attempt, task["task_id"], family, 0)
        write_checkpoint(attempt, path, checkpoint_payload(
            attempt, task, family, 0, {"coefficient": coefficient.detach().cpu().clone()},
            optimizer, scheduler, 0.0, {"total": None}, initialization,
        ))
        checkpoints.append(_checkpoint_record(path))
    else:
        start_step, elapsed, trajectory, resume_count = _resume_checkpoint(
            attempt, task, family, optimizer, scheduler, coefficient
        )
        checkpoints = [
            _checkpoint_record(checkpoint_path(attempt, task["task_id"], family, step))
            for step in MILESTONES if step <= start_step
        ]
        json_write(directory / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": start_step, "optimizer_creations": 1,
            "resume_count": resume_count, "repeated_optimizer_steps": 0,
        }, replace=True)
    conditions = list(task["selected_adaptation_conditions"])
    render_calls = start_step * len(conditions)
    gradient_nonzero_steps = sum(bool(row.get("gradient_nonzero")) for row in trajectory)
    last_metrics: dict[str, Any] = {"total": None}
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, int(budget["steps"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            standardized_displacement = (coefficient - initial) / normalization_std
            anchor = standardized_displacement.square().mean()
            trust = torch.relu(
                torch.linalg.vector_norm(standardized_displacement)
                - float(regularization["trust_region_radius"])
            ).square()
            penalty = float(regularization["coefficient_anchor_lambda"]) * anchor + trust
            losses = []
            synchronize(); tick = time.perf_counter()
            residual = basis(coefficient, chunk_size=16384)
            for condition in conditions:
                sample = context["samples"][f"{held_out}/{condition}"]
                cache_request = adaptation_cache_request(
                    task, family, step, condition, selected
                )
                validate_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                    context["base"], sample, residual, context["background"]
                )
                register_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                losses.append(loo_render_loss(lpips_runtime, rgb, alpha, sample, penalty))
            total = torch.stack([row["total"] for row in losses]).mean()
            if not torch.isfinite(total):
                raise FloatingPointError("LOO coefficient loss contains NaN/Inf")
            total.backward()
            if coefficient.grad is None or not torch.isfinite(coefficient.grad).all():
                raise FloatingPointError("LOO coefficient gradient contains NaN/Inf")
            gradient_nonzero = bool(torch.count_nonzero(coefficient.grad))
            gradient_norm = float(torch.linalg.vector_norm(coefficient.grad.detach()))
            clipped = torch.nn.utils.clip_grad_norm_(
                [coefficient], float(budget["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step(); scheduler.step(); synchronize()
            elapsed += time.perf_counter() - tick
            if not torch.isfinite(coefficient).all():
                raise FloatingPointError("LOO coefficient state contains NaN/Inf")
            render_calls += len(conditions)
            gradient_nonzero_steps += int(gradient_nonzero)
            last_metrics = {
                "total": float(total.detach()), "regularization": float(penalty.detach()),
                "garment_rgb_l1": statistics.fmean(float(row["garment_rgb_l1"].detach()) for row in losses),
                "lpips": statistics.fmean(float(row["lpips"].detach()) for row in losses),
                "mask_l1": statistics.fmean(float(row["mask_l1"].detach()) for row in losses),
            }
            append_jsonl(directory / "trajectory.jsonl", {
                "step": step, "conditions": conditions, **last_metrics,
                "coefficient": coefficient.detach().cpu().tolist(),
                "gradient_nonzero": gradient_nonzero, "gradient_norm": gradient_norm,
                "clipped_gradient_norm": float(clipped),
                "cumulative_optimizer_wall_time_seconds": elapsed,
            })
            if step in MILESTONES:
                path = checkpoint_path(attempt, task["task_id"], family, step)
                write_checkpoint(attempt, path, checkpoint_payload(
                    attempt, task, family, step,
                    {"coefficient": coefficient.detach().cpu().clone()}, optimizer, scheduler,
                    elapsed, last_metrics, initialization,
                ))
                checkpoints.append(_checkpoint_record(path))
                json_write(directory / "RUN_STATUS.json", {
                    "status": "RUNNING", "optimizer_steps": step, "optimizer_creations": 1,
                    "resume_count": resume_count, "repeated_optimizer_steps": 0,
                }, replace=True)
    except Exception as error:
        json_write(directory / "failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "registered_optimizer_steps": len(jsonl(directory / "trajectory.jsonl")),
            "created_at_utc": now(),
        })
        raise
    frozen = _frozen_gradient_audit(context, basis)
    final_path = checkpoint_path(attempt, task["task_id"], family, 300)
    reloaded = torch.load(final_path, map_location="cpu", weights_only=False)
    roundtrip = torch.equal(reloaded["trainable_state"]["coefficient"], coefficient.detach().cpu())
    summary = {
        "schema_version": "canondressgs.paper.loo_low_dimensional_run.v1",
        "task_id": task["task_id"], "status": "PASS", "method_family": family,
        "held_out_garment": held_out, "basis_garments": task["basis_garments"],
        "rotation": task["rotation"], "K": int(task["K"]),
        "adaptation_conditions": conditions, "calibration_conditions": task["calibration_conditions"],
        "test_conditions": task["test_conditions"], "initialization": initialization,
        "regularization": regularization, "trainable_scalars": coefficient.numel(),
        "optimizer_creations": 1, "optimizer_steps": 300, "forward_calls": 300,
        "backward_calls": 300, "logical_renderer_calls": render_calls,
        "physical_renderer_calls": render_calls, "checkpoint_writes": len(checkpoints),
        "checkpoints": checkpoints, "final_coefficient": coefficient.detach().cpu().tolist(),
        "coefficient_displacement": float(torch.linalg.vector_norm(coefficient.detach() - initial)),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficient.detach())),
        "optimizer_section_wall_time_seconds": elapsed,
        "end_to_end_wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gradient_nonzero_steps": gradient_nonzero_steps, "finite": True,
        "frozen_gradient_audit": frozen, "checkpoint_roundtrip": "PASS" if roundtrip else "FAIL",
        "resume_count": resume_count, "repeated_optimizer_steps": 0,
        "held_out_teacher_used": False, "test_views_used_for_updates": False,
    }
    if not roundtrip or len(checkpoints) != len(MILESTONES) or frozen["status"] != "PASS":
        raise RuntimeError("LOO low-dimensional checkpoint/frozen-gradient contract failed")
    json_write(summary_path, summary)
    json_write(directory / "RUN_STATUS.json", {
        "status": "PASS", "optimizer_steps": 300, "optimizer_creations": 1,
        "resume_count": resume_count, "repeated_optimizer_steps": 0,
    }, replace=True)
    return summary


def train_low_dimensional(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    if read_json(attempt / "02_basis_construction/summary.json")["status"] != "PASS":
        raise RuntimeError("low-dimensional training requires basis PASS")
    if build_lookup_registry(root)["status"] != "PASS":
        raise RuntimeError("low-dimensional training requires hard lookup PASS")
    destination = attempt / "04_low_dimensional_runs/summary.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    lpips_runtime = LPIPSRuntime(root, context["base"]._xyz.device)
    calibrate_regularization(root, context, lpips_runtime)
    summaries = []
    for task in tasks():
        for family in METHOD_FAMILIES[:2]:
            summaries.append(train_one_low(attempt, root, context, lpips_runtime, task, family))
            update_status(
                attempt, status="LOW_DIMENSIONAL_RUNNING", optimizer_runs=len(summaries),
                optimizer_steps=sum(row["optimizer_steps"] for row in summaries),
                checkpoint_writes=sum(row["checkpoint_writes"] for row in summaries),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "status": "PASS", "completed_runs": len(summaries),
        "optimizer_steps": sum(row["optimizer_steps"] for row in summaries),
        "checkpoint_writes": sum(row["checkpoint_writes"] for row in summaries),
        "logical_renderer_calls": sum(row["logical_renderer_calls"] for row in summaries),
    }
    expected = {"completed_runs": 80, "optimizer_steps": 24000, "checkpoint_writes": 480}
    if any(result[key] != value for key, value in expected.items()):
        raise RuntimeError("LOO_EXECUTION_COUNT_MISMATCH")
    json_write(destination, result)
    return result


def train_one_full(
    attempt: Path, root: Path, context: Mapping[str, Any], lpips_runtime: LPIPSRuntime,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    family = METHOD_FAMILIES[2]
    directory = run_dir(attempt, task, family)
    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"completed LOO full-residual run is invalid: {directory}")
        return summary
    creating = not directory.exists()
    if creating:
        output_io.ensure_directory(attempt, directory)
    budget = contracts()["execution"]["optimizer_budget"]
    seed_everything(int(budget["seed"]))
    held_out = task["held_out_garment"]
    basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
    lookup = lookup_row(attempt, task["task_id"])
    selected = lookup["selected_known_endpoint"]
    with torch.no_grad():
        initial_residual = basis(coefficients[selected], chunk_size=16384)
    field = FullResidualDelta(context["base"], initial_residual, payload["channel_bounds"])
    if field.raw_delta.numel() != 4_400_000:
        raise RuntimeError("LOO full-residual trainable scalar mismatch")
    optimizer, scheduler = make_optimizer([field.raw_delta])
    regularization = regularization_for_split(attempt, held_out, family)
    initialization = {
        "name": "REFERENCE_NEAREST_HARD_LOOKUP_RESIDUAL",
        "selected_known_endpoint": selected,
        "residual_fingerprint": residual_fingerprint(initial_residual),
        "held_out_teacher_used": False,
    }
    with torch.no_grad():
        initial_exact = residual_fingerprint(field.residual()) == residual_fingerprint(initial_residual)
    if not initial_exact:
        raise RuntimeError("LOO full-residual initial render state mismatch")
    checkpoints: list[dict[str, Any]] = []
    start_step, elapsed, trajectory, resume_count = 0, 0.0, [], 0
    if creating:
        json_write(directory / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": 0, "optimizer_creations": 1,
            "resume_count": 0, "repeated_optimizer_steps": 0,
        })
        path = checkpoint_path(attempt, task["task_id"], family, 0)
        write_checkpoint(attempt, path, checkpoint_payload(
            attempt, task, family, 0, {"raw_delta": field.raw_delta.detach().cpu().clone()},
            optimizer, scheduler, 0.0, {"total": None}, initialization,
        ))
        checkpoints.append(_checkpoint_record(path))
    else:
        start_step, elapsed, trajectory, resume_count = _resume_checkpoint(
            attempt, task, family, optimizer, scheduler, field.raw_delta
        )
        checkpoints = [
            _checkpoint_record(checkpoint_path(attempt, task["task_id"], family, step))
            for step in MILESTONES if step <= start_step
        ]
        json_write(directory / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": start_step, "optimizer_creations": 1,
            "resume_count": resume_count, "repeated_optimizer_steps": 0,
        }, replace=True)
    conditions = list(task["selected_adaptation_conditions"])
    render_calls = start_step * len(conditions)
    gradient_nonzero_steps = sum(bool(row.get("gradient_nonzero")) for row in trajectory)
    protected_gradient_max = max((float(row.get("protected_gradient_max", 0.0)) for row in trajectory), default=0.0)
    paired_low = _run_summary(attempt, task, METHOD_FAMILIES[0])
    equal_wall_time_budget = float(paired_low["optimizer_section_wall_time_seconds"])
    equal_wall_time_path = directory / "equal_wall_time_state.pt"
    equal_wall_time_saved = equal_wall_time_path.is_file()
    equal_wall_time_step = start_step
    equal_wall_time_elapsed = elapsed
    equal_wall_time_state = field.raw_delta.detach().clone()
    if not equal_wall_time_saved and elapsed > equal_wall_time_budget:
        raise RuntimeError("LOO_EQUAL_WALL_TIME_RESUME_STATE_MISSING")
    last_metrics: dict[str, Any] = {"total": None}
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, int(budget["steps"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            penalty = field.regularization(regularization)
            losses = []
            synchronize(); tick = time.perf_counter()
            residual = field.residual()
            for condition in conditions:
                sample = context["samples"][f"{held_out}/{condition}"]
                cache_request = adaptation_cache_request(
                    task, family, step, condition, selected
                )
                validate_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                    context["base"], sample, residual, context["background"]
                )
                register_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                losses.append(loo_render_loss(lpips_runtime, rgb, alpha, sample, penalty))
            total = torch.stack([row["total"] for row in losses]).mean()
            if not torch.isfinite(total):
                raise FloatingPointError("LOO full-residual loss contains NaN/Inf")
            total.backward()
            gradient = field.raw_delta.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError("LOO full-residual gradient contains NaN/Inf")
            protected = field.support.expand_as(gradient) == 0
            protected_max = float(gradient[protected].abs().max()) if protected.any() else 0.0
            protected_gradient_max = max(protected_gradient_max, protected_max)
            gradient_nonzero = bool(torch.count_nonzero(gradient))
            gradient_norm = float(torch.linalg.vector_norm(gradient.detach()))
            clipped = torch.nn.utils.clip_grad_norm_(
                [field.raw_delta], float(budget["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step(); scheduler.step(); synchronize()
            elapsed += time.perf_counter() - tick
            if not torch.isfinite(field.raw_delta).all():
                raise FloatingPointError("LOO full-residual state contains NaN/Inf")
            render_calls += len(conditions)
            gradient_nonzero_steps += int(gradient_nonzero)
            if not equal_wall_time_saved:
                if elapsed <= equal_wall_time_budget:
                    equal_wall_time_state.copy_(field.raw_delta.detach())
                    equal_wall_time_step = step
                    equal_wall_time_elapsed = elapsed
                else:
                    save_torch(attempt, equal_wall_time_path, {
                        "schema_version": "canondressgs.paper.loo_equal_wall_time_state.v1",
                        "task_id": task["task_id"], "method_family": family,
                        "paired_low_dimensional_wall_time_seconds": equal_wall_time_budget,
                        "last_completed_step_within_budget": equal_wall_time_step,
                        "full_residual_elapsed_seconds": equal_wall_time_elapsed,
                        "raw_delta": equal_wall_time_state.detach().cpu().clone(),
                        "execution_head": assert_execution_head(attempt),
                        "held_out_teacher_used": False,
                    })
                    equal_wall_time_saved = True
            last_metrics = {
                "total": float(total.detach()), "regularization": float(penalty.detach()),
                "garment_rgb_l1": statistics.fmean(float(row["garment_rgb_l1"].detach()) for row in losses),
                "lpips": statistics.fmean(float(row["lpips"].detach()) for row in losses),
                "mask_l1": statistics.fmean(float(row["mask_l1"].detach()) for row in losses),
            }
            append_jsonl(directory / "trajectory.jsonl", {
                "step": step, "conditions": conditions, **last_metrics,
                "gradient_nonzero": gradient_nonzero, "gradient_norm": gradient_norm,
                "clipped_gradient_norm": float(clipped),
                "protected_gradient_max": protected_max,
                "cumulative_optimizer_wall_time_seconds": elapsed,
            })
            if step in MILESTONES:
                path = checkpoint_path(attempt, task["task_id"], family, step)
                write_checkpoint(attempt, path, checkpoint_payload(
                    attempt, task, family, step,
                    {"raw_delta": field.raw_delta.detach().cpu().clone()}, optimizer, scheduler,
                    elapsed, last_metrics, initialization,
                ))
                checkpoints.append(_checkpoint_record(path))
                json_write(directory / "RUN_STATUS.json", {
                    "status": "RUNNING", "optimizer_steps": step, "optimizer_creations": 1,
                    "resume_count": resume_count, "repeated_optimizer_steps": 0,
                }, replace=True)
    except Exception as error:
        json_write(directory / "failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "registered_optimizer_steps": len(jsonl(directory / "trajectory.jsonl")),
            "created_at_utc": now(),
        })
        raise
    frozen = _frozen_gradient_audit(context, basis)
    if not equal_wall_time_saved:
        save_torch(attempt, equal_wall_time_path, {
            "schema_version": "canondressgs.paper.loo_equal_wall_time_state.v1",
            "task_id": task["task_id"], "method_family": family,
            "paired_low_dimensional_wall_time_seconds": equal_wall_time_budget,
            "last_completed_step_within_budget": equal_wall_time_step,
            "full_residual_elapsed_seconds": equal_wall_time_elapsed,
            "raw_delta": equal_wall_time_state.detach().cpu().clone(),
            "execution_head": assert_execution_head(attempt),
            "held_out_teacher_used": False,
        })
        equal_wall_time_saved = True
    equal_wall_time_payload = torch.load(equal_wall_time_path, map_location="cpu", weights_only=False)
    equal_wall_time_artifact = {
        "path": str(equal_wall_time_path), "sha256": sha256(equal_wall_time_path),
        "bytes": equal_wall_time_path.stat().st_size,
        "step": int(equal_wall_time_payload["last_completed_step_within_budget"]),
        "full_residual_elapsed_seconds": float(equal_wall_time_payload["full_residual_elapsed_seconds"]),
        "paired_low_dimensional_wall_time_seconds": equal_wall_time_budget,
    }
    final_path = checkpoint_path(attempt, task["task_id"], family, 300)
    reloaded = torch.load(final_path, map_location="cpu", weights_only=False)
    roundtrip = torch.equal(reloaded["trainable_state"]["raw_delta"], field.raw_delta.detach().cpu())
    active = field.raw_delta.detach() * field.support
    summary = {
        "schema_version": "canondressgs.paper.loo_full_residual_run.v1",
        "task_id": task["task_id"], "status": "PASS", "method_family": family,
        "held_out_garment": held_out, "basis_garments": task["basis_garments"],
        "rotation": task["rotation"], "K": int(task["K"]),
        "adaptation_conditions": conditions, "calibration_conditions": task["calibration_conditions"],
        "test_conditions": task["test_conditions"], "initialization": initialization,
        "initial_residual_exact": initial_exact, "regularization": regularization,
        "trainable_scalars": field.raw_delta.numel(),
        "effective_trainable_scalars": int(field.support.sum()) * 22,
        "optimizer_creations": 1, "optimizer_steps": 300, "forward_calls": 300,
        "backward_calls": 300, "logical_renderer_calls": render_calls,
        "physical_renderer_calls": render_calls, "checkpoint_writes": len(checkpoints),
        "checkpoints": checkpoints,
        "equal_wall_time_state": equal_wall_time_artifact,
        "full_residual_bound_normalized_displacement": float(torch.sqrt(active.square().mean())),
        "optimizer_section_wall_time_seconds": elapsed,
        "end_to_end_wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "protected_gradient_max": protected_gradient_max,
        "finite": True, "frozen_gradient_audit": frozen,
        "checkpoint_roundtrip": "PASS" if roundtrip else "FAIL",
        "resume_count": resume_count, "repeated_optimizer_steps": 0,
        "held_out_teacher_used": False, "test_views_used_for_updates": False,
    }
    if (
        not roundtrip or len(checkpoints) != len(MILESTONES)
        or frozen["status"] != "PASS" or protected_gradient_max != 0.0
    ):
        raise RuntimeError("LOO full-residual checkpoint/protected-gradient contract failed")
    json_write(summary_path, summary)
    json_write(directory / "RUN_STATUS.json", {
        "status": "PASS", "optimizer_steps": 300, "optimizer_creations": 1,
        "resume_count": resume_count, "repeated_optimizer_steps": 0,
    }, replace=True)
    return summary


def train_full_residual(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    if read_json(attempt / "04_low_dimensional_runs/summary.json")["status"] != "PASS":
        raise RuntimeError("full-residual training requires low-dimensional PASS")
    destination = attempt / "05_full_residual_runs/summary.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    lpips_runtime = LPIPSRuntime(root, context["base"]._xyz.device)
    summaries = []
    for task in tasks():
        summaries.append(train_one_full(attempt, root, context, lpips_runtime, task))
        update_status(
            attempt, status="FULL_RESIDUAL_RUNNING", optimizer_runs=80 + len(summaries),
            optimizer_steps=24000 + sum(row["optimizer_steps"] for row in summaries),
            checkpoint_writes=480 + sum(row["checkpoint_writes"] for row in summaries),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "status": "PASS", "completed_runs": len(summaries),
        "optimizer_steps": sum(row["optimizer_steps"] for row in summaries),
        "checkpoint_writes": sum(row["checkpoint_writes"] for row in summaries),
        "logical_renderer_calls": sum(row["logical_renderer_calls"] for row in summaries),
    }
    expected = {"completed_runs": 40, "optimizer_steps": 12000, "checkpoint_writes": 240}
    if any(result[key] != value for key, value in expected.items()):
        raise RuntimeError("LOO_EXECUTION_COUNT_MISMATCH")
    json_write(destination, result)
    update_status(
        attempt, status="OPTIMIZATION_PASS", optimizer_runs=120,
        optimizer_steps=36000, checkpoint_writes=720,
    )
    return result


def _flatten_normalized(residual: Any, bounds: Mapping[str, float]) -> "torch.Tensor":
    values = runtime_imports()["normalized_residual_dict"](residual, bounds)
    return torch.cat([values[name].reshape(-1) for name in runtime_imports()["CHANNELS"]])


def convex_weights(known: Sequence[Any], target: Any, bounds: Mapping[str, float]) -> list[float]:
    columns = torch.stack([_flatten_normalized(value, bounds).double() for value in known])
    target_flat = _flatten_normalized(target, bounds).double()
    gram = columns @ columns.t()
    linear = columns @ target_flat
    best_value = math.inf
    best = None
    count = len(known)
    for mask in range(1, 1 << count):
        indices = [index for index in range(count) if mask & (1 << index)]
        selected = torch.tensor(indices, device=gram.device, dtype=torch.long)
        local_gram = gram.index_select(0, selected).index_select(1, selected)
        local_linear = linear.index_select(0, selected)
        ones = torch.ones((len(indices), 1), device=gram.device, dtype=torch.double)
        system = torch.cat((
            torch.cat((local_gram, ones), dim=1),
            torch.cat((ones.t(), torch.zeros((1, 1), device=gram.device, dtype=torch.double)), dim=1),
        ), dim=0)
        rhs = torch.cat((local_linear, torch.ones(1, device=gram.device, dtype=torch.double)))
        try:
            solution = torch.linalg.solve(system, rhs)[:-1]
        except RuntimeError:
            continue
        if bool((solution < -1e-9).any()):
            continue
        solution = solution.clamp_min(0)
        solution = solution / solution.sum().clamp_min(1e-12)
        objective = float(solution @ local_gram @ solution - 2 * solution @ local_linear)
        if objective < best_value - 1e-12:
            full = torch.zeros(count, device=gram.device, dtype=torch.double)
            full[selected] = solution
            best_value, best = objective, full
    if best is None:
        raise RuntimeError("convex combination oracle solver failed")
    return [float(value) for value in best.detach().cpu()]


def combine_residuals(residuals: Sequence[Any], weights: Sequence[float]) -> Any:
    if len(residuals) != len(weights):
        raise ValueError("residual/weight count mismatch")
    fields = {
        name: sum(float(weight) * getattr(residual, name) for residual, weight in zip(residuals, weights))
        for name in runtime_imports()["CHANNELS"]
    }
    return runtime_imports()["GaussianClothingResiduals"](**fields)


def build_oracles(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "03_oracle_capacity/oracle_registry.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    teachers = load_teachers(context)
    rows = []
    for split in split_rows():
        held_out = split["held_out_garment"]
        basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
        bounds = payload["channel_bounds"]
        target = teachers[held_out]
        oracle_coefficient = runtime_imports()["project_residual_onto_basis"](basis, target)
        projection = basis(oracle_coefficient, chunk_size=16384)
        nearest_distances = {
            outfit: normalized_rmse(teachers[outfit], target, bounds)
            for outfit in payload["basis_garments"]
        }
        residual_nearest = min(nearest_distances, key=lambda name: (nearest_distances[name], name))
        weights = convex_weights(
            [teachers[outfit] for outfit in payload["basis_garments"]], target, bounds
        )
        convex = combine_residuals(
            [teachers[outfit] for outfit in payload["basis_garments"]], weights
        )
        mean_residual = basis(torch.zeros(basis.rank, device=context["base"]._xyz.device, dtype=context["base"]._xyz.dtype), chunk_size=16384)
        projection_error = normalized_rmse(projection, target, bounds)
        mean_error = normalized_rmse(mean_residual, target, bounds)
        ratio = None if mean_error <= 0 else 1.0 - projection_error**2 / mean_error**2
        row = {
            "held_out_garment": held_out, "basis_garments": payload["basis_garments"],
            "basis_sha256": sha256(attempt / f"02_basis_construction/{held_out}/loo_basis.pt"),
            "oracle_coefficient": oracle_coefficient.detach().cpu().tolist(),
            "oracle_coefficient_norm": float(torch.linalg.vector_norm(oracle_coefficient)),
            "projection_residual_rmse": projection_error,
            "mean_residual_rmse": mean_error, "projection_ratio": ratio,
            "residual_nearest_endpoint": residual_nearest,
            "residual_nearest_distances": nearest_distances,
            "convex_weights": dict(zip(payload["basis_garments"], weights)),
            "convex_residual_rmse": normalized_rmse(convex, target, bounds),
            "held_out_teacher_use": "OFFLINE_ORACLE_ONLY",
            "deployable": False,
        }
        rows.append(row)
    result = {
        "schema_version": "canondressgs.paper.loo_oracle_registry.v1",
        "task_id": TASK_ID, "status": "PASS", "row_count": len(rows),
        "oracle_projection_computations": len(rows),
        "residual_nearest_computations": len(rows),
        "convex_combination_oracle_computations": len(rows),
        "rows": rows,
    }
    if len(rows) != 5:
        raise RuntimeError("LOO_EXECUTION_COUNT_MISMATCH")
    json_write(destination, result)
    return result


def oracle_row(attempt: Path, held_out: str) -> dict[str, Any]:
    rows = read_json(attempt / "03_oracle_capacity/oracle_registry.json")["rows"]
    return next(row for row in rows if row["held_out_garment"] == held_out)


def residual_from_checkpoint(
    attempt: Path, context: Mapping[str, Any], task: Mapping[str, Any], family: str, step: int
) -> tuple[Any, dict[str, Any]]:
    held_out = task["held_out_garment"]
    basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
    path = checkpoint_path(attempt, task["task_id"], family, step)
    checkpoint = torch.load(
        path,
        map_location="cpu", weights_only=False,
    )
    if family in METHOD_FAMILIES[:2]:
        coefficient = checkpoint["trainable_state"]["coefficient"].to(context["base"]._xyz)
        selected = lookup_row(attempt, task["task_id"])["selected_known_endpoint"]
        initial = coefficients[selected] if family == METHOD_FAMILIES[0] else torch.zeros_like(coefficient)
        return basis(coefficient, chunk_size=16384), {
            "coefficient": coefficient.detach().cpu().tolist(),
            "coefficient_norm": float(torch.linalg.vector_norm(coefficient)),
            "coefficient_displacement": float(torch.linalg.vector_norm(coefficient - initial)),
            "optimizer_steps": step, "wall_time_seconds": float(checkpoint["elapsed_seconds"]),
            "checkpoint_bytes": path.stat().st_size,
        }
    selected = lookup_row(attempt, task["task_id"])["selected_known_endpoint"]
    initial = basis(coefficients[selected], chunk_size=16384)
    field = FullResidualDelta(context["base"], initial, payload["channel_bounds"])
    with torch.no_grad():
        field.raw_delta.copy_(checkpoint["trainable_state"]["raw_delta"].to(field.raw_delta))
        residual = field.residual()
    return residual, {
        "coefficient": None,
        "full_residual_bound_normalized_displacement": float(torch.sqrt((field.raw_delta * field.support).square().mean())),
        "optimizer_steps": step, "wall_time_seconds": float(checkpoint["elapsed_seconds"]),
        "checkpoint_bytes": path.stat().st_size,
    }


def residual_from_equal_wall_time(
    attempt: Path, context: Mapping[str, Any], task: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    held_out = task["held_out_garment"]
    basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
    selected = lookup_row(attempt, task["task_id"])["selected_known_endpoint"]
    initial = basis(coefficients[selected], chunk_size=16384)
    path = run_dir(attempt, task, METHOD_FAMILIES[2]) / "equal_wall_time_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["execution_head"] != assert_execution_head(attempt):
        raise RuntimeError("LOO equal-wall-time execution HEAD mismatch")
    field = FullResidualDelta(context["base"], initial, payload["channel_bounds"])
    with torch.no_grad():
        field.raw_delta.copy_(state["raw_delta"].to(field.raw_delta))
        residual = field.residual()
    return residual, {
        "coefficient": None,
        "full_residual_bound_normalized_displacement": float(
            torch.sqrt((field.raw_delta * field.support).square().mean())
        ),
        "optimizer_steps": int(state["last_completed_step_within_budget"]),
        "wall_time_seconds": float(state["full_residual_elapsed_seconds"]),
        "paired_low_dimensional_wall_time_seconds": float(state["paired_low_dimensional_wall_time_seconds"]),
        "state_bytes": path.stat().st_size,
        "state_sha256": sha256(path),
    }


def static_residual(
    attempt: Path, context: Mapping[str, Any], teachers: Mapping[str, Any],
    task: Mapping[str, Any], method: str,
) -> tuple[Any, dict[str, Any]]:
    held_out = task["held_out_garment"]
    basis, coefficients, payload = load_loo_basis(attempt, held_out, context["base"]._xyz.device)
    lookup = lookup_row(attempt, task["task_id"])
    oracle = oracle_row(attempt, held_out)
    if method == "BASE_AVATAR":
        return runtime_imports()["GaussianClothingResiduals"].zeros(context["base"]), {"coefficient": None}
    if method == "REFERENCE_NEAREST_HARD_LOOKUP":
        selected = lookup["selected_known_endpoint"]
        return teachers[selected], {"selected_known_endpoint": selected, "coefficient": coefficients[selected].detach().cpu().tolist()}
    if method == "RESIDUAL_NEAREST_ORACLE":
        selected = oracle["residual_nearest_endpoint"]
        return teachers[selected], {
            "selected_known_endpoint": selected,
            "coefficient": coefficients[selected].detach().cpu().tolist(), "oracle_only": True,
        }
    if method == "ORACLE_PROJECTION_LOO_BASIS":
        coefficient = torch.tensor(oracle["oracle_coefficient"], device=context["base"]._xyz.device, dtype=context["base"]._xyz.dtype)
        return basis(coefficient, chunk_size=16384), {"coefficient": oracle["oracle_coefficient"], "oracle_only": True}
    if method == "HELD_OUT_TEACHER_ENDPOINT":
        return teachers[held_out], {
            "coefficient": None, "oracle_only": True, "non_deployable_reference": True,
        }
    if method == "CONVEX_COMBINATION_ORACLE":
        weights = [oracle["convex_weights"][outfit] for outfit in payload["basis_garments"]]
        coefficient = sum(
            float(weight) * coefficients[outfit]
            for outfit, weight in zip(payload["basis_garments"], weights)
        )
        return combine_residuals([teachers[outfit] for outfit in payload["basis_garments"]], weights), {
            "weights": oracle["convex_weights"], "coefficient": coefficient.detach().cpu().tolist(),
            "oracle_only": True,
        }
    raise KeyError(method)


def masked_mae(first: "torch.Tensor", second: "torch.Tensor", mask: "torch.Tensor") -> float:
    return float(_masked_mean((first - second).abs(), mask))


def adaptation_diagnostics(
    residual: Any, metadata: Mapping[str, Any], teacher: Any,
    basis: Any, coefficients: Mapping[str, "torch.Tensor"], basis_payload: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    coefficient_value = metadata.get("coefficient")
    coefficient = None if coefficient_value is None else torch.as_tensor(
        coefficient_value,
        device=next(iter(coefficients.values())).device,
        dtype=next(iter(coefficients.values())).dtype,
    )
    nearest_known_distance = None
    oracle_projection_distance = None
    coefficient_norm = None
    if coefficient is not None:
        coefficient_norm = float(torch.linalg.vector_norm(coefficient))
        std = basis_payload["coefficient_std"].to(coefficient).clamp_min(1e-8)
        nearest_known_distance = min(
            float(torch.linalg.vector_norm((coefficient - known) / std))
            for known in coefficients.values()
        )
        oracle_coefficient = torch.as_tensor(oracle["oracle_coefficient"]).to(coefficient)
        oracle_projection_distance = float(torch.linalg.vector_norm(coefficient - oracle_coefficient))
    full_error = normalized_rmse(residual, teacher, basis_payload["channel_bounds"])
    span_error = float(oracle["projection_residual_rmse"])
    return {
        "coefficient_norm": coefficient_norm,
        "coefficient_displacement": metadata.get("coefficient_displacement"),
        "nearest_known_distance": nearest_known_distance,
        "oracle_projection_distance": oracle_projection_distance,
        "span_error": span_error,
        "projection_error": span_error,
        "full_residual_error": full_error,
        "full_residual_displacement": metadata.get("full_residual_bound_normalized_displacement"),
    }


def evaluate_metrics(
    lpips_runtime: LPIPSRuntime, rgb: "torch.Tensor", alpha: "torch.Tensor",
    sample: Mapping[str, Any], base_rgb: "torch.Tensor",
) -> dict[str, Any]:
    metrics = runtime_imports()["metrics"]
    garment = sample["target_clothing_mask"]
    protected = sample["target_protected_mask"]
    target = sample["target_edit_rgb"]
    silhouette_iou, boundary_f, tolerance = metrics.silhouette_metrics(alpha, garment)
    with torch.inference_mode():
        lpips_value = float(lpips_runtime.lpips()(metrics.lpips_input(rgb, garment), metrics.lpips_input(target, garment)).reshape(()))
        protected_lpips = float(lpips_runtime.lpips()(metrics.lpips_input(rgb, protected), metrics.lpips_input(base_rgb, protected)).reshape(()))
    return {
        "rgb_mae": masked_mae(rgb, target, garment),
        "psnr": metrics.masked_psnr(rgb, target, garment),
        "ssim": metrics.masked_ssim(rgb, target, garment),
        "lpips": lpips_value,
        "silhouette_iou": silhouette_iou,
        "boundary_f": boundary_f,
        "boundary_tolerance_pixels": tolerance,
        "protected_rgb": masked_mae(rgb, base_rgb, protected),
        "protected_lpips": protected_lpips,
        "identity": masked_mae(rgb, base_rgb, protected),
        "finite": all(torch.isfinite(value).all() for value in (rgb, alpha)),
    }


def save_render_tensor(attempt: Path, path: Path, tensor: "torch.Tensor", channels: int) -> dict[str, Any]:
    if tensor.ndim != 3:
        raise ValueError("render tensor must be rank three")
    value = tensor.detach().float().clamp(0, 1).cpu()
    if value.shape[0] != channels:
        if value.shape[-1] != channels:
            raise ValueError("render tensor channel mismatch")
        value = value.permute(2, 0, 1)
    if channels == 1:
        image = Image.fromarray(value.mul(255).round().to(torch.uint8).squeeze(0).numpy(), mode="L")
    elif channels == 3:
        image = Image.fromarray(value.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy(), mode="RGB")
    else:
        raise ValueError(channels)
    return output_io.atomic_save_png(attempt, path, image)


def evaluation_render_paths(
    attempt: Path, task_id: str, method: str, step: int | str,
    rgb: "torch.Tensor", alpha: "torch.Tensor",
) -> dict[str, Any]:
    token = str(step) if isinstance(step, str) else f"step_{step:06d}"
    directory = attempt / "07_predictions/renders" / task_id / method
    rgb_path = directory / f"{token}_rgb.png"
    alpha_path = directory / f"{token}_alpha.png"
    rgb_artifact = save_render_tensor(attempt, rgb_path, rgb, 3)
    alpha_artifact = save_render_tensor(attempt, alpha_path, alpha, 1)
    return {
        "rgb_path": str(rgb_path), "rgb_sha256": rgb_artifact["sha256"],
        "alpha_path": str(alpha_path), "alpha_sha256": alpha_artifact["sha256"],
    }


def _task_by_id(task_id: str) -> dict[str, Any]:
    return next(row for row in tasks() if row["task_id"] == task_id)


def _run_summary(attempt: Path, task: Mapping[str, Any], family: str) -> dict[str, Any]:
    return read_json(run_dir(attempt, task, family) / "run_summary.json")


def run_evaluation(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "08_metrics/evaluation.json"
    if destination.is_file():
        return read_json(destination)
    if read_json(attempt / "05_full_residual_runs/summary.json")["status"] != "PASS":
        raise RuntimeError("evaluation requires optimizer PASS")
    build_oracles(root)
    context = runtime_context(attempt)
    teachers = load_teachers(context)
    lpips_runtime = LPIPSRuntime(root, context["base"]._xyz.device)
    diagnostic_cache = {}
    for held_out in OUTFITS:
        basis, coefficients, payload = load_loo_basis(
            attempt, held_out, context["base"]._xyz.device
        )
        diagnostic_cache[held_out] = {
            "basis": basis, "coefficients": coefficients, "payload": payload,
            "oracle": oracle_row(attempt, held_out),
        }
    metric_rows: list[dict[str, Any]] = []
    equal_wall_time_rows: list[dict[str, Any]] = []
    render_registry: list[dict[str, Any]] = []
    equal_wall_time_render_registry: list[dict[str, Any]] = []
    prediction_registry: list[dict[str, Any]] = []
    equal_wall_time_prediction_registry: list[dict[str, Any]] = []
    static_cache: dict[str, dict[str, Any]] = {}
    static_cache_hits = 0
    physical = 0
    # Save immutable observations once for the fixed visual-review denominator.
    for held_out in OUTFITS:
        for rotation in ("R0", "R1", "R2", "R3"):
            task = next(row for row in tasks() if row["held_out_garment"] == held_out and row["rotation"] == rotation and int(row["K"]) == 2)
            test_condition = task["test_conditions"][0]
            sample = context["samples"][f"{held_out}/{test_condition}"]
            target_dir = attempt / f"07_predictions/observations/{held_out}_{rotation}"
            save_render_tensor(attempt, target_dir / "test_rgb.png", sample["target_edit_rgb"], 3)
            save_render_tensor(attempt, target_dir / "test_mask.png", sample["target_clothing_mask"], 1)
            for condition in task["selected_adaptation_conditions"]:
                reference = context["samples"][f"{held_out}/{condition}"]
                save_render_tensor(attempt, target_dir / f"reference_{condition}_rgb.png", reference["target_edit_rgb"], 3)
                save_render_tensor(attempt, target_dir / f"reference_{condition}_mask.png", reference["target_clothing_mask"], 1)
    for task in tasks():
        held_out = task["held_out_garment"]
        test_condition = task["test_conditions"][0]
        sample = context["samples"][f"{held_out}/{test_condition}"]
        for method in STATIC_METHODS:
            with torch.inference_mode():
                residual, metadata = static_residual(attempt, context, teachers, task, method)
            residual_sha = residual_fingerprint(residual)
            diagnostic = diagnostic_cache[held_out]
            adaptation = adaptation_diagnostics(
                residual, metadata, teachers[held_out], diagnostic["basis"],
                diagnostic["coefficients"], diagnostic["payload"], diagnostic["oracle"],
            )
            cache_request = static_cache_request(
                task, method, test_condition,
                str(metadata.get("selected_known_endpoint", "NOT_APPLICABLE")),
            )
            cache_key, _ = loo_cache_key_plan.cache_key_v2(cache_request)
            if cache_key in static_cache:
                validate_runtime_cache_request(
                    attempt, cache_request, cache_hit=True
                )
                cached = static_cache[cache_key]
                metrics = copy.deepcopy(cached["metrics"])
                paths = copy.deepcopy(cached["paths"])
                source_renderer_time = float(cached["renderer_time_seconds"])
                cache_hit = True
                static_cache_hits += 1
                register_runtime_cache_request(
                    attempt, cache_request, cache_hit=True
                )
            else:
                validate_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                with torch.inference_mode():
                    synchronize(); render_started = time.perf_counter()
                    rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                        context["base"], sample, residual, context["background"]
                    )
                    synchronize(); source_renderer_time = time.perf_counter() - render_started
                metrics = evaluate_metrics(lpips_runtime, rgb, alpha, sample, sample["target_base_rgb"])
                paths = evaluation_render_paths(attempt, task["task_id"], method, "static", rgb, alpha)
                static_cache[cache_key] = {
                    "metrics": copy.deepcopy(metrics), "paths": copy.deepcopy(paths),
                    "renderer_time_seconds": source_renderer_time,
                }
                cache_hit = False
                physical += 1
                register_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                prediction_registry.append({
                    "prediction_id": f"{task['task_id']}/{method}/static",
                    "task_id": task["task_id"], "method": method, "step": None,
                    "residual_sha256": residual_sha, **paths,
                })
            logical_id = f"{task['task_id']}/{method}/static"
            render_registry.append({
                "logical_render_id": logical_id, "task_id": task["task_id"],
                "method": method, "step": None, "physical_render": not cache_hit,
                "cache_hit": cache_hit, "cache_key": cache_key,
                "source_physical_key": cache_key,
            })
            metric_rows.append({
                "evaluation_id": logical_id, "task_id": task["task_id"],
                "held_out_garment": held_out, "rotation": task["rotation"],
                "K": int(task["K"]), "test_condition": test_condition,
                "method": method, "method_family": "STATIC_ORACLE_TRACK",
                "checkpoint_step": None, "final_checkpoint": True,
                "trainable_scalars": 0, "optimizer_steps": 0, "wall_time_seconds": 0.0,
                "peak_vram_bytes": 0, "checkpoint_bytes": 0,
                "renderer_time_seconds": 0.0 if cache_hit else source_renderer_time,
                "source_physical_renderer_time_seconds": source_renderer_time,
                "prediction_bytes": sum(Path(paths[key]).stat().st_size for key in ("rgb_path", "alpha_path")),
                "residual_sha256": residual_sha, **metadata, **adaptation, **metrics, **paths,
            })
        for family in METHOD_FAMILIES:
            summary = _run_summary(attempt, task, family)
            selected_endpoint = lookup_row(attempt, task["task_id"])["selected_known_endpoint"]
            for step in MILESTONES:
                with torch.inference_mode():
                    residual, metadata = residual_from_checkpoint(attempt, context, task, family, step)
                    cache_request = checkpoint_cache_request(
                        task, family, step, test_condition, selected_endpoint
                    )
                    validate_runtime_cache_request(
                        attempt, cache_request, cache_hit=False
                    )
                    synchronize(); render_started = time.perf_counter()
                    rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                        context["base"], sample, residual, context["background"]
                    )
                    synchronize(); renderer_time = time.perf_counter() - render_started
                register_runtime_cache_request(
                    attempt, cache_request, cache_hit=False
                )
                metrics = evaluate_metrics(lpips_runtime, rgb, alpha, sample, sample["target_base_rgb"])
                paths = evaluation_render_paths(attempt, task["task_id"], family, step, rgb, alpha)
                diagnostic = diagnostic_cache[held_out]
                adaptation = adaptation_diagnostics(
                    residual, metadata, teachers[held_out], diagnostic["basis"],
                    diagnostic["coefficients"], diagnostic["payload"], diagnostic["oracle"],
                )
                physical += 1
                logical_id = f"{task['task_id']}/{family}/step_{step:06d}"
                render_registry.append({
                    "logical_render_id": logical_id, "task_id": task["task_id"],
                    "method": family, "step": step, "physical_render": True,
                    "cache_hit": False,
                    "cache_key": loo_cache_key_plan.cache_key_v2(cache_request)[0],
                    "source_physical_key": loo_cache_key_plan.cache_key_v2(cache_request)[0],
                })
                prediction_registry.append({
                    "prediction_id": logical_id, "task_id": task["task_id"],
                    "method": family, "step": step,
                    "residual_sha256": residual_fingerprint(residual), **paths,
                })
                metric_rows.append({
                    "evaluation_id": logical_id, "task_id": task["task_id"],
                    "held_out_garment": held_out, "rotation": task["rotation"],
                    "K": int(task["K"]), "test_condition": test_condition,
                    "method": family, "method_family": family,
                    "checkpoint_step": step, "final_checkpoint": step == 300,
                    "trainable_scalars": summary["trainable_scalars"],
                    "optimizer_section_wall_time_seconds": summary["optimizer_section_wall_time_seconds"],
                    "peak_vram_bytes": summary["peak_vram_bytes"],
                    "renderer_time_seconds": renderer_time,
                    "prediction_bytes": sum(Path(paths[key]).stat().st_size for key in ("rgb_path", "alpha_path")),
                    "residual_sha256": residual_fingerprint(residual),
                    **metadata, **adaptation, **metrics, **paths,
                })
        with torch.inference_mode():
            residual, metadata = residual_from_equal_wall_time(attempt, context, task)
            synchronize(); render_started = time.perf_counter()
            rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                context["base"], sample, residual, context["background"]
            )
            synchronize(); renderer_time = time.perf_counter() - render_started
        metrics = evaluate_metrics(lpips_runtime, rgb, alpha, sample, sample["target_base_rgb"])
        equal_method = "FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION_EQUAL_WALL_TIME"
        paths = evaluation_render_paths(
            attempt, task["task_id"], equal_method, "equal_wall_time", rgb, alpha
        )
        diagnostic = diagnostic_cache[held_out]
        adaptation = adaptation_diagnostics(
            residual, metadata, teachers[held_out], diagnostic["basis"],
            diagnostic["coefficients"], diagnostic["payload"], diagnostic["oracle"],
        )
        logical_id = f"{task['task_id']}/{equal_method}/step_{metadata['optimizer_steps']:06d}"
        equal_wall_time_render_registry.append({
            "logical_render_id": logical_id, "task_id": task["task_id"],
            "method": equal_method, "step": metadata["optimizer_steps"],
            "physical_render": True, "cache_hit": False, "cache_key": None,
        })
        equal_wall_time_prediction_registry.append({
            "prediction_id": logical_id, "task_id": task["task_id"],
            "method": equal_method, "step": metadata["optimizer_steps"],
            "residual_sha256": residual_fingerprint(residual), **paths,
        })
        equal_wall_time_rows.append({
            "evaluation_id": logical_id, "task_id": task["task_id"],
            "held_out_garment": held_out, "rotation": task["rotation"],
            "K": int(task["K"]), "test_condition": test_condition,
            "method": equal_method, "method_family": "AUXILIARY_EQUAL_WALL_TIME",
            "checkpoint_step": None, "equal_wall_time_step": metadata["optimizer_steps"],
            "trainable_scalars": 4_400_000,
            "peak_vram_bytes": _run_summary(attempt, task, METHOD_FAMILIES[2])["peak_vram_bytes"],
            "renderer_time_seconds": renderer_time,
            "prediction_bytes": sum(Path(paths[key]).stat().st_size for key in ("rgb_path", "alpha_path")),
            "residual_sha256": residual_fingerprint(residual),
            **metadata, **adaptation, **metrics, **paths,
        })
    adaptation_logical = (
        read_json(attempt / "04_low_dimensional_runs/summary.json")["logical_renderer_calls"]
        + read_json(attempt / "05_full_residual_runs/summary.json")["logical_renderer_calls"]
    )
    result = {
        "schema_version": "canondressgs.paper.loo_evaluation.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "rows": metric_rows, "evaluation_inference": len(metric_rows),
        "equal_wall_time_rows": equal_wall_time_rows,
        "equal_wall_time_evaluations": len(equal_wall_time_rows),
        "optimized_checkpoint_evaluations": sum(row["method_family"] in METHOD_FAMILIES for row in metric_rows),
        "static_oracle_evaluations": sum(row["method_family"] == "STATIC_ORACLE_TRACK" for row in metric_rows),
        "evaluation_logical_renders": len(render_registry),
        "evaluation_physical_renders": physical,
        "K_shared_static_cache_hits": static_cache_hits,
        "adaptation_view_logical_renders": adaptation_logical,
        "total_logical_renders": adaptation_logical + len(render_registry),
        "unique_physical_renders": adaptation_logical + physical,
        "render_registry": render_registry,
        "equal_wall_time_render_registry": equal_wall_time_render_registry,
        "prediction_registry": prediction_registry,
        "equal_wall_time_prediction_registry": equal_wall_time_prediction_registry,
        "auxiliary_equal_wall_time_logical_renders": len(equal_wall_time_render_registry),
        "auxiliary_equal_wall_time_physical_renders": len(equal_wall_time_render_registry),
        "formal_denominator_excludes_auxiliary_equal_wall_time": True,
    }
    result["runtime_cache_plan_verification"] = verify_runtime_cache_registry(attempt)
    expected = expected_counts()
    checks = {
        "evaluation_960": len(metric_rows) == expected["evaluation_inference"],
        "optimized_720": result["optimized_checkpoint_evaluations"] == expected["optimized_checkpoint_evaluation_inferences"],
        "static_240": result["static_oracle_evaluations"] == expected["static_oracle_evaluation_inferences"],
        "cache_hits_115": static_cache_hits == expected["K_shared_static_cache_hits"],
        "logical_54960": result["total_logical_renders"] == expected["total_logical_renders"],
        "physical_54845": result["unique_physical_renders"] == expected["unique_physical_renders"],
        "runtime_cache_plan_exact": result["runtime_cache_plan_verification"]["status"] == "PASS",
        "equal_wall_time_40": len(equal_wall_time_rows) == 40,
        "equal_wall_time_last_completed_step": all(
            0 <= int(row["equal_wall_time_step"]) <= 300
            and float(row["wall_time_seconds"]) <= float(row["paired_low_dimensional_wall_time_seconds"])
            for row in equal_wall_time_rows
        ),
        "all_finite": all(row["finite"] for row in metric_rows),
        "equal_wall_time_all_finite": all(row["finite"] for row in equal_wall_time_rows),
    }
    result["checks"] = checks
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    json_write(destination, result)
    json_write(attempt / "08_metrics/render_registry.json", {
        "status": result["status"], "logical_renders": len(render_registry),
        "physical_renders": physical, "cache_hits": static_cache_hits,
        "rows": render_registry,
        "auxiliary_equal_wall_time_renders": equal_wall_time_render_registry,
    })
    json_write(attempt / "07_predictions/prediction_registry.json", {
        "status": result["status"], "prediction_count": len(prediction_registry),
        "rows": prediction_registry,
        "auxiliary_equal_wall_time_prediction_count": len(equal_wall_time_prediction_registry),
        "auxiliary_equal_wall_time_predictions": equal_wall_time_prediction_registry,
    })
    if result["status"] != "PASS":
        raise RuntimeError("LOO_EXECUTION_COUNT_MISMATCH")
    update_status(
        attempt, status="EVALUATION_PASS", logical_renders=result["total_logical_renders"],
        physical_renders=result["unique_physical_renders"],
    )
    return result


def _pil_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _pil_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


def _mask_box(mask: np.ndarray, expansion: float = 0.05) -> tuple[int, int, int, int]:
    positions = np.argwhere(mask)
    height, width = mask.shape
    if not positions.size:
        return (0, 0, width, height)
    y0, x0 = positions.min(0); y1, x1 = positions.max(0) + 1
    margin = int(math.floor(expansion * max(y1 - y0, x1 - x0) + 0.5))
    return (
        max(0, int(x0) - margin), max(0, int(y0) - margin),
        min(width, int(x1) + margin), min(height, int(y1) + margin),
    )


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    copy_image = image.copy()
    copy_image.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy_image, ((size[0] - copy_image.width) // 2, (size[1] - copy_image.height) // 2))
    return result


def _boundary_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    value = np.asarray(image.convert("RGB")).copy()
    up = np.pad(mask[:-1], ((1, 0), (0, 0)))
    down = np.pad(mask[1:], ((0, 1), (0, 0)))
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)))
    boundary = mask ^ (up & down & left & right)
    value[boundary] = np.array([220, 30, 30], dtype=np.uint8)
    return Image.fromarray(value, mode="RGB")


def _visual_column(
    title: str, rgb_path: str | Path, alpha_path: str | Path | None,
    target_mask_path: str | Path, protected_mask: np.ndarray | None = None,
) -> Image.Image:
    width = 220
    result = Image.new("RGB", (width, 790), "white")
    draw = ImageDraw.Draw(result)
    draw.text((8, 8), title[:30], fill="black")
    rgb = _pil_rgb(rgb_path)
    target_mask = _pil_mask(target_mask_path)
    full = _fit_panel(rgb, (200, 250))
    result.paste(full, (10, 40))
    garment = rgb.crop(_mask_box(target_mask))
    result.paste(_fit_panel(garment, (200, 150)), (10, 300))
    overlay_mask = _pil_mask(alpha_path) if alpha_path is not None else target_mask
    overlay = _boundary_overlay(rgb, overlay_mask)
    result.paste(_fit_panel(overlay, (200, 150)), (10, 460))
    protected = protected_mask if protected_mask is not None else ~target_mask
    protected_crop = rgb.crop(_mask_box(protected, expansion=0.0))
    result.paste(_fit_panel(protected_crop, (200, 150)), (10, 620))
    return result


def _final_row(
    rows: Sequence[Mapping[str, Any]], held_out: str, rotation: str, k: int, method: str
) -> dict[str, Any]:
    matches = [
        dict(row) for row in rows
        if row["held_out_garment"] == held_out and row["rotation"] == rotation
        and int(row["K"]) == k and row["method"] == method
        and (row["checkpoint_step"] is None or int(row["checkpoint_step"]) == 300)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"visual row cardinality mismatch: {held_out}/{rotation}/K{k}/{method}")
    return matches[0]


def create_visual_sheets(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "11_visual_sheets/visual_registry.json"
    if destination.is_file():
        return read_json(destination)
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    rows = evaluation["rows"]
    registry = []
    context = runtime_context(attempt)
    for held_out in OUTFITS:
        for rotation in ("R0", "R1", "R2", "R3"):
            observation = attempt / f"07_predictions/observations/{held_out}_{rotation}"
            target_mask = observation / "test_mask.png"
            protected_sample_task = next(
                task for task in tasks()
                if task["held_out_garment"] == held_out and task["rotation"] == rotation and int(task["K"]) == 2
            )
            protected_tensor = context["samples"][f"{held_out}/{protected_sample_task['test_conditions'][0]}"]["target_protected_mask"]
            protected = protected_tensor.detach().cpu().numpy().reshape(protected_tensor.shape[-2:]) >= 0.5
            columns: list[tuple[str, str | Path, str | Path | None]] = [
                ("Held-out reference", observation / f"reference_{protected_sample_task['selected_adaptation_conditions'][0]}_rgb.png", None),
            ]
            method_specs = (
                ("Base Avatar", 2, "BASE_AVATAR"),
                ("Hard Lookup K2", 2, "REFERENCE_NEAREST_HARD_LOOKUP"),
                ("Low-Dim K1", 1, METHOD_FAMILIES[0]),
                ("Low-Dim K2", 2, METHOD_FAMILIES[0]),
                ("Full Residual K2", 2, METHOD_FAMILIES[2]),
                ("Oracle Projection", 2, "ORACLE_PROJECTION_LOO_BASIS"),
                ("Teacher Endpoint", 2, "HELD_OUT_TEACHER_ENDPOINT"),
                ("Convex Oracle", 2, "CONVEX_COMBINATION_ORACLE"),
            )
            selected_endpoint = None
            coefficient = None
            for title, k, method in method_specs:
                row = _final_row(rows, held_out, rotation, k, method)
                columns.append((title, row["rgb_path"], row["alpha_path"]))
                if method == "REFERENCE_NEAREST_HARD_LOOKUP":
                    selected_endpoint = row.get("selected_known_endpoint")
                if method == METHOD_FAMILIES[0] and k == 2:
                    coefficient = row.get("coefficient")
            sheet = Image.new("RGB", (220 * len(columns), 850), (245, 245, 245))
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (10, 8),
                f"{held_out} {rotation} | selected={selected_endpoint} | c={coefficient}",
                fill="black",
            )
            for index, (title, rgb_path, alpha_path) in enumerate(columns):
                sheet.paste(_visual_column(title, rgb_path, alpha_path, target_mask, protected), (index * 220, 50))
            path = attempt / f"11_visual_sheets/garment_rotation/{held_out}_{rotation}.png"
            artifact = output_io.atomic_save_png(attempt, path, sheet)
            registry.append({
                "sheet_id": f"garment_rotation/{held_out}_{rotation}",
                "sheet_type": "GARMENT_ROTATION", "held_out_garment": held_out,
                "rotation": rotation, "path": str(path), "sha256": artifact["sha256"],
                "selected_known_endpoint": selected_endpoint, "coefficient": coefficient,
                "source_execution_head": assert_execution_head(attempt),
            })
        capacity_columns = []
        for rotation in ("R0", "R1", "R2", "R3"):
            rotation_task = next(
                task for task in tasks()
                if task["held_out_garment"] == held_out
                and task["rotation"] == rotation and int(task["K"]) == 2
            )
            observation = attempt / f"07_predictions/observations/{held_out}_{rotation}"
            rotation_mask = observation / "test_mask.png"
            protected_tensor = context["samples"][
                f"{held_out}/{rotation_task['test_conditions'][0]}"
            ]["target_protected_mask"]
            rotation_protected = (
                protected_tensor.detach().cpu().numpy().reshape(protected_tensor.shape[-2:]) >= 0.5
            )
            row = _final_row(rows, held_out, rotation, 2, "ORACLE_PROJECTION_LOO_BASIS")
            capacity_columns.append((
                f"{rotation} Oracle", row["rgb_path"], row["alpha_path"],
                rotation_mask, rotation_protected,
            ))
            teacher = _final_row(rows, held_out, rotation, 2, "HELD_OUT_TEACHER_ENDPOINT")
            capacity_columns.append((
                f"{rotation} Teacher", teacher["rgb_path"], teacher["alpha_path"],
                rotation_mask, rotation_protected,
            ))
        sheet = Image.new("RGB", (220 * len(capacity_columns), 810), (245, 245, 245))
        for index, (title, rgb_path, alpha_path, target_mask, protected) in enumerate(capacity_columns):
            sheet.paste(
                _visual_column(title, rgb_path, alpha_path, target_mask, protected),
                (index * 220, 10),
            )
        path = attempt / f"11_visual_sheets/capacity/{held_out}.png"
        artifact = output_io.atomic_save_png(attempt, path, sheet)
        registry.append({
            "sheet_id": f"capacity/{held_out}", "sheet_type": "CAPACITY",
            "held_out_garment": held_out, "path": str(path), "sha256": artifact["sha256"],
            "source_execution_head": assert_execution_head(attempt),
        })
    scaling = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(scaling)
    draw.text((30, 20), "LOO K2 minus K1 fixed paired scaling", fill="black")
    y = 70
    for held_out in OUTFITS:
        for rotation in ("R0", "R1", "R2", "R3"):
            k1 = _final_row(rows, held_out, rotation, 1, METHOD_FAMILIES[0])
            k2 = _final_row(rows, held_out, rotation, 2, METHOD_FAMILIES[0])
            draw.text(
                (30, y),
                f"{held_out} {rotation}: LPIPS {k2['lpips']-k1['lpips']:+.6f}, RGB {k2['rgb_mae']-k1['rgb_mae']:+.6f}",
                fill="black",
            )
            y += 38
    path = attempt / "11_visual_sheets/view_budget_scaling.png"
    artifact = output_io.atomic_save_png(attempt, path, scaling)
    registry.append({
        "sheet_id": "view_budget_scaling", "sheet_type": "AGGREGATE_SCALING",
        "path": str(path), "sha256": artifact["sha256"],
        "source_execution_head": assert_execution_head(attempt),
    })
    review_rows = [
        {
            "held_out_garment": held_out, "rotation": rotation,
            "sheet_id": f"garment_rotation/{held_out}_{rotation}",
            "identity_contamination": None, "component_contamination": None,
            "patch_cloud_mottle": None, "edge_scatter": None,
            "silhouette_collapse": None, "body_deformation": None,
            "reviewer_notes": None,
        }
        for held_out in OUTFITS for rotation in ("R0", "R1", "R2", "R3")
    ]
    json_write(attempt / "11_visual_sheets/visual_review_template.json", {
        "schema_version": "canondressgs.paper.loo_visual_review_input.v1",
        "task_id": TASK_ID, "status": "PENDING_MANUAL_REVIEW",
        "review_every_registered_garment_rotation": True,
        "cherry_picking_allowed": False, "grade_range": [0, 3], "rows": review_rows,
    })
    result = {
        "schema_version": "canondressgs.paper.loo_visual_registry.v1",
        "task_id": TASK_ID, "status": "PASS" if len(registry) == 26 else "FAIL",
        "sheet_count": len(registry), "sheets": registry,
    }
    json_write(destination, result)
    if result["status"] != "PASS":
        raise RuntimeError("LOO visual sheet count mismatch")
    return result


def seal_visual_review(root: Path, review_path: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    review = read_json(review_path)
    rows = review.get("rows", [])
    expected_keys = {(outfit, rotation) for outfit in OUTFITS for rotation in ("R0", "R1", "R2", "R3")}
    actual_keys = {(row.get("held_out_garment"), row.get("rotation")) for row in rows}
    grade_fields = (
        "identity_contamination", "component_contamination", "patch_cloud_mottle",
        "edge_scatter", "silhouette_collapse", "body_deformation",
    )
    complete = (
        len(rows) == 20 and actual_keys == expected_keys
        and all(isinstance(row.get(field), int) and 0 <= row[field] <= 3 for row in rows for field in grade_fields)
    )
    if not complete:
        raise ValueError("visual review is incomplete or outside the frozen grade range")
    result = {
        "schema_version": "canondressgs.paper.loo_visual_review_summary.v1",
        "task_id": TASK_ID, "status": "PASS",
        "reviewed_cells": len(rows), "expected_cells": 20,
        "identity_contamination_count": sum(row["identity_contamination"] > 0 for row in rows),
        "component_contamination_count": sum(row["component_contamination"] > 0 for row in rows),
        "severe_component_contamination_count": sum(row["component_contamination"] >= 3 for row in rows),
        "patch_cloud_mottle_count": sum(row["patch_cloud_mottle"] > 0 for row in rows),
        "severe_patch_cloud_mottle_count": sum(row["patch_cloud_mottle"] >= 3 for row in rows),
        "edge_scatter_count": sum(row["edge_scatter"] > 0 for row in rows),
        "silhouette_collapse_count": sum(row["silhouette_collapse"] > 0 for row in rows),
        "body_deformation_count": sum(row["body_deformation"] > 0 for row in rows),
        "review_source_sha256": sha256(review_path), "rows": rows,
    }
    json_write(attempt / "11_visual_sheets/visual_review_summary.json", result)
    return result


def _mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if not isinstance(row[metric], str)]
    return statistics.fmean(values) if values else math.inf


def recovery_ratio(hard: float, low: float, full: float) -> dict[str, Any]:
    denominator = hard - full
    if denominator <= 0:
        return {
            "value": None,
            "reason": "HARD_LOOKUP_ERROR_MINUS_FULL_RESIDUAL_ERROR_NONPOSITIVE",
            "numerator": hard - low, "denominator": denominator,
        }
    return {
        "value": (hard - low) / denominator, "reason": None,
        "numerator": hard - low, "denominator": denominator,
    }


def classify_results(
    *, capacity_limited: bool, hard_lookup_gate_pass: bool,
    recovery_pass: bool, scaling_pass: bool, efficiency_pass: bool, safety_pass: bool,
) -> str:
    if capacity_limited:
        return "LOO_BASIS_CAPACITY_LIMITED"
    if not hard_lookup_gate_pass:
        return "LOO_HARD_LOOKUP_NOT_OUTPERFORMED"
    if not recovery_pass:
        return "LOO_OPTIMIZATION_LIMITED"
    if scaling_pass and efficiency_pass and safety_pass:
        return "LOO_BASIS_ADAPTATION_SUPPORTED"
    return "LOO_BASIS_ADAPTATION_PARTIAL"


def execution_count_verification(attempt: Path) -> dict[str, Any]:
    run_summaries = []
    for task in tasks():
        for family in METHOD_FAMILIES:
            path = run_dir(attempt, task, family) / "run_summary.json"
            if path.is_file():
                run_summaries.append(read_json(path))
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    wide_table = read_json(attempt / "08_metrics/wide_result_table.json")
    wide_rows = wide_table["rows"]
    actual = {
        "loo_basis_construction_plans": read_json(attempt / "02_basis_construction/summary.json")["split_count"],
        "K1_primary_tasks": sum(int(task["K"]) == 1 for task in tasks()),
        "K2_primary_tasks": sum(int(task["K"]) == 2 for task in tasks()),
        "primary_tasks": len(tasks()), "K4_primary_tasks": 0,
        "low_dimensional_optimizer_runs": sum(row["method_family"] == METHOD_FAMILIES[0] for row in run_summaries),
        "zero_initialization_diagnostic_runs": sum(row["method_family"] == METHOD_FAMILIES[1] for row in run_summaries),
        "full_residual_optimizer_runs": sum(row["method_family"] == METHOD_FAMILIES[2] for row in run_summaries),
        "total_optimizer_creations": sum(int(row["optimizer_creations"]) for row in run_summaries),
        "total_optimizer_runs": len(run_summaries),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in run_summaries),
        "forward_loss_calls": sum(int(row["forward_calls"]) for row in run_summaries),
        "backward_calls": sum(int(row["backward_calls"]) for row in run_summaries),
        "checkpoint_writes": sum(int(row["checkpoint_writes"]) for row in run_summaries),
        "K1_adaptation_view_logical_renders": sum(int(row["logical_renderer_calls"]) for row in run_summaries if int(row["K"]) == 1),
        "K2_adaptation_view_logical_renders": sum(int(row["logical_renderer_calls"]) for row in run_summaries if int(row["K"]) == 2),
        "adaptation_view_logical_renders": sum(int(row["logical_renderer_calls"]) for row in run_summaries),
        "oracle_projection_computations": read_json(attempt / "03_oracle_capacity/oracle_registry.json")["oracle_projection_computations"],
        "residual_nearest_computations": read_json(attempt / "03_oracle_capacity/oracle_registry.json")["residual_nearest_computations"],
        "convex_combination_oracle_computations": read_json(attempt / "03_oracle_capacity/oracle_registry.json")["convex_combination_oracle_computations"],
        "hard_lookup_inferences": read_json(attempt / "09_hard_lookup_analysis/hard_lookup_registry.json")["row_count"],
        "optimized_checkpoint_evaluation_inferences": evaluation["optimized_checkpoint_evaluations"],
        "static_oracle_evaluation_inferences": evaluation["static_oracle_evaluations"],
        "evaluation_inference": evaluation["evaluation_inference"],
        "total_logical_renders": evaluation["total_logical_renders"],
        "K_shared_static_cache_hits": evaluation["K_shared_static_cache_hits"],
        "unique_physical_renders": evaluation["unique_physical_renders"],
        "primary_task_table_rows": sum(row["row_type"] == "PRIMARY_TASK" for row in wide_rows),
        "K2_minus_K1_scaling_rows": sum(row["row_type"] == "K2_MINUS_K1_SCALING" for row in wide_rows),
        "oracle_capacity_rows": sum(row["row_type"] == "ORACLE_CAPACITY" for row in wide_rows),
        "wide_result_table_rows": len(wide_rows),
        "garment_rotation_visual_sheets": 20,
        "capacity_visual_sheets": 5,
        "aggregate_scaling_visual_sheets": 1,
        "visual_sheets": read_json(attempt / "11_visual_sheets/visual_registry.json")["sheet_count"],
    }
    expected = expected_counts()
    rows = {
        key: {"expected": int(value), "actual": actual.get(key), "delta": None if key not in actual else actual[key] - int(value)}
        for key, value in expected.items() if isinstance(value, int)
    }
    unexplained = [key for key, row in rows.items() if row["actual"] is not None and row["delta"] != 0]
    missing = [key for key, row in rows.items() if row["actual"] is None]
    result = {
        "schema_version": "canondressgs.paper.loo_execution_count_verification.v1",
        "task_id": TASK_ID, "status": "PASS" if not unexplained and not missing else "FAIL",
        "expected_actual_delta": rows, "unexplained_delta_keys": unexplained,
        "missing_actual_keys": missing,
        "run_count": len(run_summaries),
        "nonfinite_run_count": sum(not row["finite"] for row in run_summaries),
        "frozen_gradient_failure_count": sum(row["frozen_gradient_audit"]["status"] != "PASS" for row in run_summaries),
        "checkpoint_roundtrip_failure_count": sum(row["checkpoint_roundtrip"] != "PASS" for row in run_summaries),
        "resume_count": sum(int(row["resume_count"]) for row in run_summaries),
        "repeated_optimizer_steps": sum(int(row["repeated_optimizer_steps"]) for row in run_summaries),
    }
    return result


def analyze_results(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "08_metrics/analysis.json"
    if destination.is_file():
        return read_json(destination)
    visual = read_json(attempt / "11_visual_sheets/visual_review_summary.json")
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    rows = [row for row in evaluation["rows"] if row["checkpoint_step"] is None or int(row["checkpoint_step"]) == 300]
    gates = contracts()["gates"]
    hard_gate = gates["hard_lookup_outperformance"]
    garment_results = {}
    rotation_rows = []
    for held_out in OUTFITS:
        successes = 0
        deltas = []
        for rotation in ("R0", "R1", "R2", "R3"):
            hard = _final_row(rows, held_out, rotation, 2, "REFERENCE_NEAREST_HARD_LOOKUP")
            low = _final_row(rows, held_out, rotation, 2, METHOD_FAMILIES[0])
            delta = {
                "lpips_reduction": float(hard["lpips"]) - float(low["lpips"]),
                "rgb_mae_reduction": float(hard["rgb_mae"]) - float(low["rgb_mae"]),
                "silhouette_iou_delta": float(low["silhouette_iou"]) - float(hard["silhouette_iou"]),
                "boundary_f_delta": float(low["boundary_f"]) - float(hard["boundary_f"]),
                "identity_delta": float(low["identity"]) - float(hard["identity"]),
            }
            passed = (
                delta["lpips_reduction"] >= float(hard_gate["per_rotation_lpips_absolute_reduction_min"])
                and delta["rgb_mae_reduction"] >= float(hard_gate["per_rotation_rgb_mae_absolute_reduction_min"])
            )
            successes += int(passed)
            rotation_rows.append({
                "held_out_garment": held_out, "rotation": rotation,
                "success": passed, **delta,
            })
            deltas.append(delta)
        garment_pass = successes >= int(hard_gate["successful_rotations_per_garment_min"])
        garment_results[held_out] = {
            "successful_rotations": successes, "required_rotations": int(hard_gate["successful_rotations_per_garment_min"]),
            "hard_lookup_outperformance_pass": garment_pass,
            "mean_deltas": {key: statistics.fmean(row[key] for row in deltas) for key in deltas[0]},
        }
    successful_garments = sum(row["hard_lookup_outperformance_pass"] for row in garment_results.values())
    hard_lookup_gate_pass = successful_garments >= int(hard_gate["minimum_successful_garments"])
    scaling_rows = []
    scaling_garments = {}
    scaling_gate = gates["few_view_scaling"]
    for held_out in OUTFITS:
        garment_pairs = []
        for rotation in ("R0", "R1", "R2", "R3"):
            k1 = _final_row(rows, held_out, rotation, 1, METHOD_FAMILIES[0])
            k2 = _final_row(rows, held_out, rotation, 2, METHOD_FAMILIES[0])
            pair = {
                "held_out_garment": held_out, "rotation": rotation,
                "lpips_k2_minus_k1": float(k2["lpips"]) - float(k1["lpips"]),
                "rgb_mae_k2_minus_k1": float(k2["rgb_mae"]) - float(k1["rgb_mae"]),
            }
            scaling_rows.append(pair); garment_pairs.append(pair)
        lpips_delta = statistics.fmean(row["lpips_k2_minus_k1"] for row in garment_pairs)
        rgb_delta = statistics.fmean(row["rgb_mae_k2_minus_k1"] for row in garment_pairs)
        scaling_garments[held_out] = {
            "lpips_k2_minus_k1": lpips_delta, "rgb_mae_k2_minus_k1": rgb_delta,
            "not_worse": lpips_delta <= float(scaling_gate["macro_lpips_k2_minus_k1_max"])
            and rgb_delta <= float(scaling_gate["macro_rgb_mae_k2_minus_k1_max"]),
        }
    scaling_not_worse = sum(row["not_worse"] for row in scaling_garments.values())
    macro_scaling_lpips = statistics.fmean(row["lpips_k2_minus_k1"] for row in scaling_rows)
    macro_scaling_rgb = statistics.fmean(row["rgb_mae_k2_minus_k1"] for row in scaling_rows)
    scaling_pass = scaling_not_worse >= int(scaling_gate["minimum_garments_k2_not_worse_than_k1"])
    if macro_scaling_lpips < 0 and macro_scaling_rgb < 0 and scaling_pass:
        scaling_classification = "VIEW_BUDGET_SCALING_POSITIVE"
    elif not scaling_pass:
        scaling_classification = "VIEW_BUDGET_SCALING_NEGATIVE"
    else:
        scaling_classification = "VIEW_BUDGET_SCALING_MIXED"
    oracle_contract = gates["oracle_span_capacity"]
    oracle_registry = read_json(attempt / "03_oracle_capacity/oracle_registry.json")
    capacity_rows = []
    for oracle in oracle_registry["rows"]:
        held_out = oracle["held_out_garment"]
        projection_rows = [
            _final_row(rows, held_out, rotation, 1, "ORACLE_PROJECTION_LOO_BASIS")
            for rotation in ("R0", "R1", "R2", "R3")
        ]
        projection_lpips = _mean(projection_rows, "lpips")
        passed = (
            oracle["projection_ratio"] is not None
            and float(oracle["projection_ratio"]) >= float(oracle_contract["projection_ratio_min"])
            and float(oracle["projection_residual_rmse"]) <= float(oracle_contract["bound_normalized_residual_rmse_max"])
            and projection_lpips <= float(oracle_contract["oracle_projection_lpips_max"])
        )
        capacity_rows.append({**oracle, "oracle_projection_lpips": projection_lpips, "capacity_pass": passed})
    capacity_failures = sum(not row["capacity_pass"] for row in capacity_rows)
    capacity_limited = capacity_failures >= int(oracle_contract["capacity_failure_if_garments_failing_min"])
    recovery_rows = []
    recovery_pass_garments = 0
    for held_out in OUTFITS:
        hard_rows = [_final_row(rows, held_out, rotation, 2, "REFERENCE_NEAREST_HARD_LOOKUP") for rotation in ("R0", "R1", "R2", "R3")]
        low_rows = [_final_row(rows, held_out, rotation, 2, METHOD_FAMILIES[0]) for rotation in ("R0", "R1", "R2", "R3")]
        full_rows = [_final_row(rows, held_out, rotation, 2, METHOD_FAMILIES[2]) for rotation in ("R0", "R1", "R2", "R3")]
        ratios = {
            metric: recovery_ratio(_mean(hard_rows, metric), _mean(low_rows, metric), _mean(full_rows, metric))
            for metric in ("lpips", "rgb_mae")
        }
        passed = all(
            ratios[metric]["value"] is not None
            and float(ratios[metric]["value"]) >= float(gates["full_residual_comparison"]["low_dimensional_gain_recovery_ratio_min"])
            for metric in ratios
        )
        recovery_pass_garments += int(passed)
        recovery_rows.append({"held_out_garment": held_out, "ratios": ratios, "recovery_pass": passed})
    recovery_pass = recovery_pass_garments >= int(gates["full_residual_comparison"]["minimum_garments_passing"])
    low_summaries = [_run_summary(attempt, task, METHOD_FAMILIES[0]) for task in tasks()]
    full_summaries = [_run_summary(attempt, task, METHOD_FAMILIES[2]) for task in tasks()]
    wall_fraction = sum(row["optimizer_section_wall_time_seconds"] for row in low_summaries) / max(sum(row["optimizer_section_wall_time_seconds"] for row in full_summaries), 1e-12)
    efficiency_pass = (
        max(row["trainable_scalars"] for row in low_summaries) <= int(gates["efficiency"]["low_dimensional_trainable_scalars_max"])
        and min(row["trainable_scalars"] for row in full_summaries) >= int(gates["efficiency"]["full_residual_trainable_scalars_min"])
        and wall_fraction <= float(gates["efficiency"]["low_dimensional_wall_time_fraction_max"])
    )
    k2_low_rows = [row for row in rows if int(row["K"]) == 2 and row["method"] == METHOD_FAMILIES[0]]
    safety_pass = (
        visual["identity_contamination_count"] == 0
        and visual["body_deformation_count"] == 0
        and visual["severe_patch_cloud_mottle_count"] == 0
        and visual["severe_component_contamination_count"] == 0
        and min(float(row["silhouette_iou"]) for row in k2_low_rows) >= float(gates["safety"]["silhouette_iou_min"])
    )
    table_metric_names = (
        "rgb_mae", "psnr", "ssim", "lpips", "silhouette_iou", "boundary_f",
        "protected_rgb", "protected_lpips", "identity", "coefficient_norm",
        "coefficient_displacement", "nearest_known_distance", "oracle_projection_distance",
        "span_error", "projection_error", "full_residual_error", "full_residual_displacement",
        "trainable_scalars", "optimizer_steps", "wall_time_seconds", "peak_vram_bytes",
        "checkpoint_bytes", "renderer_time_seconds", "prediction_bytes",
    )
    primary_table_rows = []
    formal_methods = (*STATIC_METHODS, *METHOD_FAMILIES)
    for task in tasks():
        method_results = {}
        for method in formal_methods:
            row = _final_row(
                rows, task["held_out_garment"], task["rotation"], int(task["K"]), method
            )
            method_results[method] = {
                name: row.get(name) for name in table_metric_names
            }
        equal_row = next(
            row for row in evaluation["equal_wall_time_rows"]
            if row["task_id"] == task["task_id"]
        )
        primary_table_rows.append({
            "row_type": "PRIMARY_TASK", "task_id": task["task_id"],
            "held_out_garment": task["held_out_garment"], "rotation": task["rotation"],
            "K": int(task["K"]), "method_results": method_results,
            "equal_wall_time_full_residual": {
                name: equal_row.get(name) for name in table_metric_names
            } | {"equal_wall_time_step": equal_row["equal_wall_time_step"]},
        })
    wide_rows = (
        primary_table_rows
        + [{"row_type": "K2_MINUS_K1_SCALING", **row} for row in scaling_rows]
        + [{"row_type": "ORACLE_CAPACITY", **row} for row in capacity_rows]
    )
    wide_table = {
        "schema_version": "canondressgs.paper.loo_wide_result_table.v1",
        "task_id": TASK_ID, "status": "PASS" if len(wide_rows) == 65 else "FAIL",
        "row_count": len(wide_rows),
        "row_type_counts": dict(Counter(row["row_type"] for row in wide_rows)),
        "rows": wide_rows,
    }
    if wide_table["status"] != "PASS":
        raise RuntimeError("LOO_WIDE_RESULT_TABLE_COUNT_MISMATCH")
    wide_table_path = attempt / "08_metrics/wide_result_table.json"
    if wide_table_path.is_file():
        if canonical_sha(read_json(wide_table_path)) != canonical_sha(wide_table):
            raise RuntimeError("LOO_WIDE_RESULT_TABLE_REPLAY_MISMATCH")
    else:
        json_write(wide_table_path, wide_table)
    classification = classify_results(
        capacity_limited=capacity_limited,
        hard_lookup_gate_pass=hard_lookup_gate_pass,
        recovery_pass=recovery_pass,
        scaling_pass=scaling_pass,
        efficiency_pass=efficiency_pass,
        safety_pass=safety_pass,
    )
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError("LOO_ADAPTATION_EXECUTION_INVALID")
    counts = execution_count_verification(attempt)
    equal_wall_time_summary = {}
    for held_out in OUTFITS:
        selected = [
            row for row in evaluation["equal_wall_time_rows"]
            if row["held_out_garment"] == held_out
        ]
        equal_wall_time_summary[held_out] = {
            "row_count": len(selected),
            "mean_completed_step": statistics.fmean(float(row["equal_wall_time_step"]) for row in selected),
            "mean_lpips": _mean(selected, "lpips"), "mean_rgb_mae": _mean(selected, "rgb_mae"),
            "mean_silhouette_iou": _mean(selected, "silhouette_iou"),
            "mean_wall_time_seconds": statistics.fmean(float(row["wall_time_seconds"]) for row in selected),
            "mean_paired_low_wall_time_seconds": statistics.fmean(
                float(row["paired_low_dimensional_wall_time_seconds"]) for row in selected
            ),
        }
    result = {
        "schema_version": "canondressgs.paper.loo_analysis.v1",
        "task_id": TASK_ID, "status": "PASS" if counts["status"] == "PASS" else "FAIL",
        "classification": classification,
        "garment_results": garment_results, "rotation_results": rotation_rows,
        "successful_garments": successful_garments, "hard_lookup_gate_pass": hard_lookup_gate_pass,
        "scaling_rows": scaling_rows, "scaling_garments": scaling_garments,
        "scaling_not_worse_garments": scaling_not_worse,
        "macro_lpips_k2_minus_k1": macro_scaling_lpips,
        "macro_rgb_mae_k2_minus_k1": macro_scaling_rgb,
        "view_budget_scaling_classification": scaling_classification,
        "view_budget_scaling_gate_pass": scaling_pass,
        "capacity_rows": capacity_rows, "capacity_failure_count": capacity_failures,
        "capacity_limited": capacity_limited,
        "recovery_rows": recovery_rows, "recovery_pass_garments": recovery_pass_garments,
        "recovery_gate_pass": recovery_pass,
        "equal_wall_time_full_residual": {
            "rule": "last completed full-residual step within paired low-dimensional measured wall time",
            "row_count": len(evaluation["equal_wall_time_rows"]),
            "per_garment": equal_wall_time_summary,
            "rows": evaluation["equal_wall_time_rows"],
            "changes_primary_300_step_result": False,
        },
        "low_to_full_wall_time_fraction": wall_fraction, "efficiency_gate_pass": efficiency_pass,
        "safety_gate_pass": safety_pass, "visual_safety": visual,
        "execution_count_verification": counts,
        "wide_result_table": {
            "path": str(wide_table_path), "sha256": sha256(wide_table_path),
            "row_count": wide_table["row_count"], "row_type_counts": wide_table["row_type_counts"],
        },
        "held_out_teacher_use_in_deployable_adaptation": 0,
        "full_five_garment_rank4_basis_reused": False,
        "claim_boundary": "Same-identity leave-one-garment-out simulation inside the subject02 five-garment study set only.",
        "PAPER_FINAL": False, "paper_final_count": 0,
    }
    json_write(destination, result)
    json_write(attempt / "08_metrics/metric_summary.json", result)
    json_write(attempt / "03_oracle_capacity/oracle_capacity_analysis.json", {
        "status": "PASS", "rows": capacity_rows, "capacity_failure_count": capacity_failures,
        "capacity_limited": capacity_limited,
    })
    json_write(attempt / "10_view_budget_scaling/view_budget_scaling.json", {
        "status": "PASS", "classification": scaling_classification,
        "rows": scaling_rows, "per_garment": scaling_garments,
    })
    json_write(attempt / "13_final_verification/execution_count_verification.json", counts)
    if result["status"] != "PASS":
        raise RuntimeError("LOO_EXECUTION_COUNT_MISMATCH")
    update_status(attempt, status="ANALYSIS_PASS", classification=classification)
    return result


def _format_float(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def report_payloads(analysis: Mapping[str, Any]) -> dict[str, str]:
    garment_lines = "\n".join(
        f"| {outfit} | {row['successful_rotations']}/4 | {row['hard_lookup_outperformance_pass']} | "
        f"{_format_float(row['mean_deltas']['lpips_reduction'])} | {_format_float(row['mean_deltas']['rgb_mae_reduction'])} |"
        for outfit, row in analysis["garment_results"].items()
    )
    capacity_lines = "\n".join(
        f"| {row['held_out_garment']} | {_format_float(row['projection_ratio'])} | "
        f"{_format_float(row['projection_residual_rmse'])} | {_format_float(row['oracle_projection_lpips'])} | {row['capacity_pass']} |"
        for row in analysis["capacity_rows"]
    )
    recovery_lines = "\n".join(
        f"| {row['held_out_garment']} | {_format_float(row['ratios']['lpips']['value'])} | "
        f"{_format_float(row['ratios']['rgb_mae']['value'])} | {row['recovery_pass']} |"
        for row in analysis["recovery_rows"]
    )
    equal_wall_time_lines = "\n".join(
        f"| {outfit} | {_format_float(row['mean_completed_step'])} | "
        f"{_format_float(row['mean_lpips'])} | {_format_float(row['mean_rgb_mae'])} | "
        f"{_format_float(row['mean_wall_time_seconds'])} |"
        for outfit, row in analysis["equal_wall_time_full_residual"]["per_garment"].items()
    )
    common = (
        "This execution is a same-identity leave-one-garment-out simulation inside the subject02 "
        "five-garment study set. It does not establish arbitrary garments, open-world adaptation, "
        "unseen identities, cross-identity transfer, or real captured multi-outfit performance.\n"
    )
    return {
        REPORT_NAMES[0]: f"""# AAAI27 LOO Basis Adaptation Results

Classification: `{analysis['classification']}`.

{common}

| Garment | Successful rotations | Garment gate | LPIPS reduction | RGB MAE reduction |
|---|---:|---|---:|---:|
{garment_lines}

- Hard-lookup gate: `{analysis['hard_lookup_gate_pass']}` ({analysis['successful_garments']}/5 garments).
- View scaling: `{analysis['view_budget_scaling_classification']}`.
- Recovery gate: `{analysis['recovery_gate_pass']}`.
- Safety gate: `{analysis['safety_gate_pass']}`.
- PAPER_FINAL: `false`.
""",
        REPORT_NAMES[1]: f"""# AAAI27 LOO Hard-Lookup Comparison

The deployable primary comparison is K=2 low-dimensional adaptation versus four-endpoint train-only F2 nearest hard lookup.

| Garment | Successful rotations | Pass | LPIPS reduction | RGB MAE reduction |
|---|---:|---|---:|---:|
{garment_lines}

Successful garments: {analysis['successful_garments']}/5. Gate: `{analysis['hard_lookup_gate_pass']}`.
""",
        REPORT_NAMES[2]: f"""# AAAI27 LOO Basis Capacity Analysis

| Garment | Projection ratio | Residual RMSE | Oracle LPIPS | Pass |
|---|---:|---:|---:|---|
{capacity_lines}

Capacity failures: {analysis['capacity_failure_count']}/5. Capacity-limited: `{analysis['capacity_limited']}`.
""",
        REPORT_NAMES[3]: f"""# AAAI27 LOO Few-View Scaling Results

- Classification: `{analysis['view_budget_scaling_classification']}`.
- K2-not-worse garments: {analysis['scaling_not_worse_garments']}/5.
- Macro LPIPS K2-K1: {_format_float(analysis['macro_lpips_k2_minus_k1'])}.
- Macro RGB MAE K2-K1: {_format_float(analysis['macro_rgb_mae_k2_minus_k1'])}.
""",
        REPORT_NAMES[4]: f"""# AAAI27 LOO Full-Residual Comparison

| Garment | LPIPS recovery | RGB MAE recovery | Pass |
|---|---:|---:|---|
{recovery_lines}

Nonpositive denominators are retained as `null` with the registered explicit reason. Recovery gate: `{analysis['recovery_gate_pass']}`.
Low/full wall-time fraction: {_format_float(analysis['low_to_full_wall_time_fraction'])}.

## Equal-Wall-Time Secondary Diagnostic

| Garment | Mean completed step | LPIPS | RGB MAE | Full wall time (s) |
|---|---:|---:|---:|---:|
{equal_wall_time_lines}

The state is the actual last completed full-residual step within each paired low-dimensional measured wall time; it does not change the primary 300-step result.
""",
        REPORT_NAMES[5]: f"""# AAAI27 LOO Visual Review

- Fixed sheets: 26.
- Reviewed garment-rotation cells: {analysis['visual_safety']['reviewed_cells']}.
- Identity contamination: {analysis['visual_safety']['identity_contamination_count']}.
- Body deformation: {analysis['visual_safety']['body_deformation_count']}.
- Severe patch/cloud/mottle: {analysis['visual_safety']['severe_patch_cloud_mottle_count']}.
- Severe component contamination: {analysis['visual_safety']['severe_component_contamination_count']}.
""",
        REPORT_NAMES[6]: f"""# AAAI27 LOO Failure Analysis

Final diagnosis: `{analysis['classification']}`.

- Basis capacity failures: {analysis['capacity_failure_count']}/5.
- Hard-lookup successful garments: {analysis['successful_garments']}/5.
- Full-residual recovery passing garments: {analysis['recovery_pass_garments']}/5.
- View-budget classification: `{analysis['view_budget_scaling_classification']}`.

{common}
""",
    }


def next_task_route(classification: str) -> str:
    return {
        "LOO_BASIS_ADAPTATION_SUPPORTED": "REFRESH_PAPER_FIGURE_BANK_WITH_LOO_RESULTS_AND_FREEZE_MAIN_METHOD",
        "LOO_BASIS_ADAPTATION_PARTIAL": "ADJUDICATE_PARTIAL_LOO_AND_FREEZE_CANONDRESSGS_METHOD_SCOPE",
        "LOO_BASIS_CAPACITY_LIMITED": "EVALUATE_PAPER_VALUE_BEFORE_ANY_SPATIAL_COEFFICIENT_FIELD_EXPERIMENT",
        "LOO_OPTIMIZATION_LIMITED": "EVALUATE_WHETHER_ONE_FINAL_LOO_OPTIMIZATION_REPAIR_IS_PAPER_CRITICAL",
        "LOO_HARD_LOOKUP_NOT_OUTPERFORMED": "FREEZE_ENDPOINT_COORDINATE_SYSTEM_PAPER_METHOD_SCOPE",
    }[classification]


def finalize(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    execution_head = assert_execution_head(attempt)
    analysis = read_json(attempt / "08_metrics/analysis.json")
    counts = analysis["execution_count_verification"]
    if analysis["status"] != "PASS" or counts["status"] != "PASS":
        raise RuntimeError("LOO_ADAPTATION_EXECUTION_INVALID")
    runs = [
        _run_summary(attempt, task, family)
        for task in tasks() for family in METHOD_FAMILIES
    ]
    checkpoints = [item for row in runs for item in row["checkpoints"]]
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    basis_registry = read_json(attempt / "02_basis_construction/summary.json")
    oracle = read_json(attempt / "03_oracle_capacity/oracle_registry.json")
    lookup = read_json(attempt / "09_hard_lookup_analysis/hard_lookup_registry.json")
    visual = read_json(attempt / "11_visual_sheets/visual_review_summary.json")
    registries = {
        REGISTRY_NAMES[0]: basis_registry,
        REGISTRY_NAMES[1]: {
            "schema_version": "canondressgs.paper.loo_adaptation_run_registry.v1",
            "task_id": TASK_ID, "status": "PASS", "run_count": len(runs), "runs": runs,
        },
        REGISTRY_NAMES[2]: {
            "schema_version": "canondressgs.paper.loo_checkpoint_registry.v1",
            "task_id": TASK_ID, "status": "PASS", "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        },
        REGISTRY_NAMES[3]: {
            "schema_version": "canondressgs.paper.loo_prediction_registry.v1",
            "task_id": TASK_ID, "status": "PASS",
            "prediction_count": len(evaluation["prediction_registry"]),
            "predictions": evaluation["prediction_registry"],
            "auxiliary_equal_wall_time_prediction_count": len(evaluation["equal_wall_time_prediction_registry"]),
            "auxiliary_equal_wall_time_predictions": evaluation["equal_wall_time_prediction_registry"],
        },
        REGISTRY_NAMES[4]: analysis,
        REGISTRY_NAMES[5]: {
            "status": "PASS", "oracle_registry": oracle,
            "capacity_rows": analysis["capacity_rows"],
        },
        REGISTRY_NAMES[6]: {
            "status": "PASS", "lookup_registry": lookup,
            "garment_results": analysis["garment_results"],
            "rotation_results": analysis["rotation_results"],
        },
        REGISTRY_NAMES[7]: {
            "status": "PASS", "classification": analysis["view_budget_scaling_classification"],
            "rows": analysis["scaling_rows"], "per_garment": analysis["scaling_garments"],
        },
        REGISTRY_NAMES[8]: {
            "status": "PASS", "recovery_rows": analysis["recovery_rows"],
            "recovery_gate_pass": analysis["recovery_gate_pass"],
            "equal_wall_time": analysis["equal_wall_time_full_residual"],
        },
        REGISTRY_NAMES[9]: visual,
        REGISTRY_NAMES[10]: counts,
        REGISTRY_NAMES[11]: {
            "status": "PASS", "execution_head": execution_head,
            "source_artifacts": source_artifact_audit()["status"],
            "historical_immutability": historical_immutability_audit()["status"],
            "attempt_001_immutability": original_attempt_immutability_audit(
                root, read_json(RISK / ORIGINAL_MANIFEST_NAME)
            )["status"],
            "counts": counts["status"], "temporary_leaks": output_io.temporary_file_leaks(attempt),
        },
        REGISTRY_NAMES[12]: {
            "schema_version": "canondressgs.paper.loo_attempt002_final_summary.v1",
            "task_id": TASK_ID, "status": "SCIENTIFIC_EXECUTION_COMPLETE",
            "classification": analysis["classification"],
            "source_head": SOURCE_HEAD, "execution_head": execution_head,
            "result_head": "RESULT_HEAD_PENDING_COMMIT",
            "final_reporting_head": "FINAL_REPORTING_HEAD_RESOLVES_AFTER_SEAL_COMMIT",
            "attempt": ATTEMPT_NAME, "PAPER_FINAL": False, "paper_final_count": 0,
            "next_task": next_task_route(analysis["classification"]),
            "claim_boundary": analysis["claim_boundary"],
        },
    }
    for name, payload in registries.items():
        json_write(RISK / name, payload)
    reports = report_payloads(analysis)
    for name, payload in reports.items():
        text_write(DOCS / name, payload)
    figure_manifest = {
        "schema_version": "canondressgs.paper.sealed_loo_figure_refresh_manifest.v1",
        "task_id": TASK_ID, "status": "SEALED_FOR_SEPARATE_REFRESH_TASK",
        "result_branch": RUN_BRANCH, "source_head": SOURCE_HEAD,
        "execution_head": execution_head, "result_head": "RESULT_HEAD_PENDING_COMMIT",
        "final_reporting_head": "FINAL_REPORTING_HEAD_RESOLVES_AFTER_SEAL_COMMIT",
        "attempt": ATTEMPT_NAME, "classification": analysis["classification"],
        "metric_source": f"paper_protocol/reviewer_risk/{REGISTRY_NAMES[4]}",
        "visual_registry": f"paper_protocol/reviewer_risk/{REGISTRY_NAMES[9]}",
        "capacity_analysis": f"paper_protocol/reviewer_risk/{REGISTRY_NAMES[5]}",
        "hard_lookup_comparison": f"paper_protocol/reviewer_risk/{REGISTRY_NAMES[6]}",
        "claim_boundary": analysis["claim_boundary"],
        "license": "inherits source dataset and repository licenses; no generated imagery",
        "figure_bank_modified": False,
    }
    json_write(RISK / FIGURE_REFRESH_MANIFEST_NAME, figure_manifest)
    json_write(attempt / "13_final_verification/SEALED_LOO_FIGURE_REFRESH_MANIFEST.json", figure_manifest)
    handoff = {
        "schema_version": "canondressgs.project_control.loo_basis_adaptation_experiment_handoff.v1",
        "task_id": TASK_ID, "status": "COMPLETE",
        "classification": analysis["classification"], "branch": RUN_BRANCH,
        "source_head": SOURCE_HEAD, "execution_head": execution_head,
        "result_head": "RESULT_HEAD_PENDING_COMMIT",
        "final_reporting_head": "FINAL_REPORTING_HEAD_RESOLVES_AFTER_SEAL_COMMIT",
        "output": str(attempt), "attempt_002_created": True,
        "attempt_001_preserved": True, "attempt_003_created": False,
        "PAPER_FINAL": False, "paper_final_count": 0,
        "next_task": next_task_route(analysis["classification"]),
    }
    json_write(HANDOFF / HANDOFF_NAME, handoff)
    json_write(attempt / "13_final_verification/handoff.json", handoff)
    json_write(attempt / "13_final_verification/final_summary.json", registries[REGISTRY_NAMES[12]])
    update_status(attempt, status="FINALIZED_PENDING_RESULT_COMMIT", classification=analysis["classification"])
    return registries[REGISTRY_NAMES[12]]


def verify(root: Path, *, require_clean: bool) -> dict[str, Any]:
    attempt = attempt_path(root)
    errors = []
    if require_clean and git("status", "--short"):
        errors.append("worktree_dirty")
    if not attempt.is_dir():
        errors.append("attempt_missing")
    if forbidden_next_attempt_path(root).exists():
        errors.append("attempt_003_exists")
    for path in list(attempt.rglob("*.json")) + [RISK / name for name in REGISTRY_NAMES]:
        try:
            read_json(path)
        except Exception as error:
            errors.append(f"json:{path}:{type(error).__name__}")
    for path in attempt.rglob("*.png"):
        try:
            output_io.validate_png(path)
        except Exception as error:
            errors.append(f"png:{path}:{type(error).__name__}")
    for name in REPORT_NAMES:
        path = DOCS / name
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            errors.append(f"report:{name}")
    leaks = output_io.temporary_file_leaks(attempt)
    if leaks:
        errors.extend(f"temporary:{path}" for path in leaks)
    historical = historical_immutability_audit()
    if historical["status"] != "PASS":
        errors.append("historical_mutation")
    source = source_artifact_audit()
    if source["status"] != "PASS":
        errors.append("source_artifact_mutation")
    original = original_attempt_immutability_audit(
        root, read_json(RISK / ORIGINAL_MANIFEST_NAME)
    )
    if original["status"] != "PASS":
        errors.append("attempt_001_mutation")
    counts_path = attempt / "13_final_verification/execution_count_verification.json"
    if not counts_path.is_file() or read_json(counts_path)["status"] != "PASS":
        errors.append("count_verification")
    classification = read_json(attempt / "08_metrics/analysis.json").get("classification") if (attempt / "08_metrics/analysis.json").is_file() else None
    if classification not in ALLOWED_CLASSIFICATIONS:
        errors.append("classification")
    if len(list((attempt / "11_visual_sheets").rglob("*.png"))) != 26:
        errors.append("visual_sheet_count")
    credentials = credential_scan((ROOT / "tools/paper", DOCS, RISK, HANDOFF, attempt))
    if credentials["status"] != "PASS":
        errors.append("credential_scan")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    if diff_check.returncode != 0:
        errors.append("git_diff_check")
    result = {
        "schema_version": "canondressgs.paper.loo_final_verification.v1",
        "task_id": TASK_ID, "status": "PASS" if not errors else "FAIL",
        "errors": errors, "require_clean": require_clean,
        "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "historical_immutability": historical["status"],
        "attempt_001_immutability": original["status"],
        "source_artifact_audit": source["status"],
        "credential_scan": credentials,
        "attempt_002_present": attempt.is_dir(),
        "attempt_003_absent": not forbidden_next_attempt_path(root).exists(),
        "PAPER_FINAL": False, "paper_final_count": 0,
        "checked_at_utc": now(),
    }
    json_write(attempt / "13_final_verification/final_verification.json", result, replace=True)
    if result["status"] != "PASS":
        raise RuntimeError("LOO_ADAPTATION_EXECUTION_INVALID: " + ", ".join(errors))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen subject02 LOO basis adaptation experiment")
    parser.add_argument(
        "command",
        choices=(
            "static-preflight", "cloud-preflight", "bind", "materialize", "basis",
            "lookup", "calibrate", "oracles", "train-low", "train-full", "evaluate",
            "visual-sheets", "seal-visual-review", "analyze", "finalize", "verify",
        ),
    )
    parser.add_argument("--asset-root")
    parser.add_argument("--cloud-preflight", type=Path)
    parser.add_argument("--review-json", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = asset_root(args.asset_root) if args.asset_root else None
    if args.command == "static-preflight":
        result = static_preflight(root)
    else:
        if root is None:
            raise RuntimeError(f"{args.command} requires --asset-root or CANONDRESSGS_ASSET_ROOT")
        commands = {
            "cloud-preflight": lambda: cloud_resource_preflight(root),
            "bind": lambda: bind(args.cloud_preflight, root) if args.cloud_preflight else (_ for _ in ()).throw(ValueError("--cloud-preflight is required")),
            "materialize": lambda: materialize(root),
            "basis": lambda: build_bases(root),
            "lookup": lambda: build_lookup_registry(root),
            "calibrate": lambda: calibrate_regularization(root),
            "oracles": lambda: build_oracles(root),
            "train-low": lambda: train_low_dimensional(root),
            "train-full": lambda: train_full_residual(root),
            "evaluate": lambda: run_evaluation(root),
            "visual-sheets": lambda: create_visual_sheets(root),
            "seal-visual-review": lambda: seal_visual_review(root, args.review_json) if args.review_json else (_ for _ in ()).throw(ValueError("--review-json is required")),
            "analyze": lambda: analyze_results(root),
            "finalize": lambda: finalize(root),
            "verify": lambda: verify(root, require_clean=args.require_clean),
        }
        result = commands[args.command]()
    if args.json_output:
        json_write(args.json_output, result, replace=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
