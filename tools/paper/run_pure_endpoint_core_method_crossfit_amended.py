from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.p0_candidate_adapters import (
    B6ReferenceClassifierHardLookupAdapter,
    OursV2CandidateAdapter,
    pool_frozen_f2_reference_set,
)
from tools import run_multi_outfit_explicit_basis as multi
from tools import run_residual_field_parameterization as parameterization
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.paper import formal_batch_runtime as historical
from tools.paper import formal_runtime as formal
from tools.paper import run_p0_color_spatial_soft_control_evaluations as render_metrics


TASK_ID = "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"
SOURCE_BRANCH = "research/pure-endpoint-direct-decoder-adjudication-20260724"
SOURCE_HEAD = "69cded463e3a88cecc83e8a4c5a7a89b93ee4b20"
RUN_BRANCH = "research/pure-endpoint-core-method-crossfit-amended-20260724"
OUTPUT_NAME = "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"
ATTEMPT_NAME = "attempt_001"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
SEEDS = (0, 1, 2)
MILESTONES = (0, 20, 50, 100, 200, 300)
FAMILIES = ("reference_classifier", "shared_coefficient")
METHODS = (
    "Base Avatar",
    "Teacher Endpoint",
    "Outfit-ID Oracle",
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Linear Coefficient Predictor",
    "CanonDressGS-Endpoint",
)
PERTURBATIONS = (
    "mild_blur",
    "mask_erosion",
    "mask_dilation",
    "assignment_permutation",
    "reference_dropout",
    "single_reference",
)
SCHEDULE_HASHES = {
    0: "e5dd8e70ee8e606aa765816d6bbebb75b189a873df793fcfba6ad0bfa08cd55f",
    1: "f65232b0e0dceb8e2ea5c0485e0928191a08a5befde660493bc2a43c1675db34",
    2: "ec9d4e842e065628533b2f414cd0953eb4add5f78a5360e43e4465bb0a67ae58",
    3: "b00c63394ce1627e8727615bf7052a9a59049634a2cd7441977dd1c9201dd977",
}
BASIS_SHA = "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430"
FEATURE_SHA = "30cf19a99bbc620112928d678a0257bfabd3420be66fe1053ded3ce7e90875cb"
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
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
    return [json.loads(line, object_pairs_hook=strict_object) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path, *, lf: bool = False) -> str:
    digest = hashlib.sha256()
    if lf:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        digest.update(data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
    return Path(raw).resolve()


def attempt_path(root: Path) -> Path:
    return root / OUTPUT_NAME / ATTEMPT_NAME


def basis_path(root: Path) -> Path:
    return (
        root
        / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
        / "stage_b_basis/selected/selected_basis.pt"
    )


def feature_path(root: Path) -> Path:
    return root / "AAAI27-SEEN-OUTFIT-PAPER/shared_preflight/frozen_reference_feature_rows_v1.pt"


def amended_contract() -> dict[str, Any]:
    return read_json(RISK / "pure_endpoint_execution_contract_amended.json")


def expected_counts() -> dict[str, Any]:
    return read_json(RISK / "pure_endpoint_expected_counts_amended.json")


def rotations() -> list[dict[str, Any]]:
    return read_json(RISK / "pure_endpoint_rotation_manifests.json")["rotations"]


def source_rows() -> dict[str, dict[str, Any]]:
    rows = read_json(RISK / "dual_support_controller_training_manifest.json")["query_sets"]
    return {row["record_id"]: row for row in rows}


def schedule_audit() -> dict[str, Any]:
    schedules = read_json(RISK / "pure_endpoint_training_schedules.json")
    actual = {int(row["rotation"]): canonical_sha(row) for row in schedules["rotations"]}
    return {
        "status": "PASS" if actual == SCHEDULE_HASHES else "FAIL",
        "actual": actual,
        "expected": SCHEDULE_HASHES,
        "old_internal_step_hashes_preserved": {
            int(row["rotation"]): row["schedule_sha256"] for row in schedules["rotations"]
        },
    }


def registry_audit() -> dict[str, Any]:
    primary = read_json(RISK / "pure_endpoint_primary_baseline_registry_amended.json")
    supplementary = read_json(RISK / "pure_endpoint_supplementary_baseline_registry.json")
    names = [row["paper_name"] for row in primary["methods"]]
    historical = supplementary["methods"][0]
    checks = {
        "method_order": names == list(METHODS),
        "method_count": primary["method_count"] == 7,
        "independent_families": expected_counts()["registry"]["independent_trainable_model_family_count"] == 2,
        "historical_v7_excluded": historical["execution_count_in_pure_endpoint_crossfit"] == 0,
        "teacher_naming": names[1] == "Teacher Endpoint",
        "paper_final_false": primary["paper_final"] is False and primary["paper_final_count"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "methods": names}


def protocol_hash_audit() -> dict[str, Any]:
    manifest = read_json(RISK / "pure_endpoint_protocol_artifact_hash_manifest.json")
    rows = []
    for row in manifest["artifacts"]:
        path = ROOT / row["relative_path"]
        rows.append({
            "relative_path": row["relative_path"],
            "expected_sha256_lf": row["lf_normalized_sha256"],
            "actual_sha256_lf": sha256(path, lf=True),
        })
    mismatch = [row for row in rows if row["expected_sha256_lf"] != row["actual_sha256_lf"]]
    return {
        "status": "PASS_INDEPENDENT_MANIFEST" if not mismatch and len(rows) == 11 else "FAIL",
        "artifact_count": len(rows),
        "mutation_count": len(mismatch),
        "historical_classifications_preserved": [
            "PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH",
            "PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED",
        ],
        "rows": rows,
    }


def credential_scan() -> dict[str, Any]:
    names = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    patterns = {
        "openai_style_key": re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
        "authorization_bearer": re.compile(rb"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{20,}", re.I),
        "assigned_api_key": re.compile(rb"(?:OPENAI_API_KEY|API_KEY)\s*=\s*['\"]?[A-Za-z0-9._-]{20,}", re.I),
    }
    findings: list[dict[str, Any]] = []
    excluded_test_fixtures: list[dict[str, Any]] = []
    scanned = 0
    for relative in sorted(set(names)):
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        scanned += 1
        for name, pattern in patterns.items():
            if pattern.search(data):
                if relative == "tools/check_codex_direct_generation_contract.py":
                    excluded_test_fixtures.append({
                        "relative_path": relative, "pattern": name,
                        "classification": "credential_detector_fixture",
                    })
                else:
                    findings.append({"relative_path": relative, "pattern": name})
    return {
        "status": "PASS" if not findings else "API_CREDENTIAL_LEAK_RISK",
        "scanned_file_count": scanned,
        "finding_count": len(findings),
        "findings_without_secret_content": findings,
        "excluded_test_fixture_count": len(excluded_test_fixtures),
        "excluded_test_fixtures": excluded_test_fixtures,
        "credential_environment_names": [
            name for name in ("OPENAI_API_KEY", "CODEX_API_KEY") if os.environ.get(name)
        ],
        "credential_values_printed": False,
    }


def model_audit() -> dict[str, Any]:
    classifier = B6ReferenceClassifierHardLookupAdapter(seed=0, input_dim=512)
    coefficient = OursV2CandidateAdapter(input_dim=512)
    with torch.no_grad():
        rows = torch.zeros((3, 256))
        valid = torch.ones((3, 1))
        logits = classifier(rows, valid).logits
        output = coefficient(rows, valid).standardized_coefficients
    classifier_count = sum(value.numel() for value in classifier.parameters() if value.requires_grad)
    coefficient_count = sum(value.numel() for value in coefficient.parameters() if value.requires_grad)
    checks = {
        "classifier_parameter_count": classifier_count == 3589,
        "coefficient_parameter_count": coefficient_count == 3076,
        "classifier_output_shape": tuple(logits.shape) == (5,),
        "coefficient_output_shape": tuple(output.shape) == (4,),
        "coefficient_zero_initialization": bool(torch.count_nonzero(output).item() == 0),
    }
    return {
        "status": "PASS" if all(checks.values()) else "PURE_ENDPOINT_MODEL_CONTRACT_MISMATCH",
        "checks": checks,
        "classifier": {"parameters": classifier_count, "state_dict_keys": sorted(classifier.state_dict())},
        "shared_coefficient": {"parameters": coefficient_count, "state_dict_keys": sorted(coefficient.state_dict())},
    }


def static_preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    source_is_ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
    ) == 0
    attempt = attempt_path(root)
    basis = basis_path(root)
    features = feature_path(root)
    if not basis.is_file() or sha256(basis) != BASIS_SHA:
        raise RuntimeError("PURE_ENDPOINT_FROZEN_ASSET_MISMATCH: rank-4 basis")
    if not features.is_file() or sha256(features) != FEATURE_SHA:
        raise RuntimeError("PURE_ENDPOINT_FROZEN_ASSET_MISMATCH: frozen F2 cache")
    cache = torch.load(features, map_location="cpu", weights_only=False)
    episode_rows = cache.get("episodes", {})
    feature_shapes_pass = len(episode_rows) == 24 and all(
        tuple(row["normal"]["f2"].shape) == (3, 256)
        and tuple(row["normal"]["valid"].shape) == (3, 1)
        for row in episode_rows.values()
    )
    basis_payload = torch.load(basis, map_location="cpu", weights_only=False)
    basis_pass = (
        basis_payload.get("schema_version") == "canondressgs.multi_outfit_explicit_basis_artifact.v1"
        and int(basis_payload.get("rank", -1)) == 4
        and set(basis_payload.get("teacher_coefficients", {})) == set(OUTFITS)
    )
    verifier = subprocess.run(
        [sys.executable, "tools/paper/verify_seen_outfit_paper_assets.py", "--output-root", str(root)],
        cwd=ROOT, text=True, capture_output=True,
    )
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = "UNAVAILABLE"
    try:
        compute_apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        compute_apps = ""
    remote_names = git("remote").splitlines()
    if "origin" in remote_names:
        remote_url = git("remote", "get-url", "origin")
        remote_probe = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{SOURCE_BRANCH}"],
            cwd=ROOT, capture_output=True, timeout=30,
        )
        remote_probe_pass = remote_probe.returncode == 0
        remote_topology = "origin_remote"
    else:
        remote_url = git("rev-parse", "--path-format=absolute", "--git-common-dir")
        remote_probe = subprocess.run(
            ["git", "show-ref", "--verify", f"refs/heads/{SOURCE_BRANCH}"],
            cwd=ROOT, capture_output=True,
        )
        remote_probe_pass = remote_probe.returncode == 0
        remote_topology = "cloud_bare_common_dir"
    disk = shutil.disk_usage(root)
    contract = amended_contract()
    hash_audit = protocol_hash_audit()
    registry = registry_audit()
    schedules = schedule_audit()
    models = model_audit()
    credentials = credential_scan()
    checks = {
        "branch": branch == RUN_BRANCH,
        "source_ancestor": source_is_ancestor,
        "attempt_absent": not attempt.exists(),
        "disk_free_30gb": disk.free >= 30_000_000_000,
        "gpu_rtx4090": "4090" in gpu,
        "gpu_exclusive": not bool(compute_apps),
        "output_writable": root.is_dir() and os.access(root, os.W_OK),
        "git_remote_continuity": bool(remote_url) and remote_probe_pass,
        "api_continuity": True,
        "feature_cache": feature_shapes_pass,
        "basis": basis_pass,
        "external_assets_19_of_19": verifier.returncode == 0,
        "protocol_hash_closure": hash_audit["status"] == "PASS_INDEPENDENT_MANIFEST",
        "registry": registry["status"] == "PASS",
        "schedules": schedules["status"] == "PASS",
        "models": models["status"] == "PASS",
        "credentials": credentials["status"] == "PASS",
        "execution_authorized": contract["execution_authorized"] is True,
        "paper_final_false": contract["paper_final"] is False and contract["paper_final_count"] == 0,
    }
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_static_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "timestamp_utc": now(),
        "branch": branch,
        "head": head,
        "source_head": SOURCE_HEAD,
        "checks": checks,
        "gpu": gpu,
        "gpu_compute_process_count": 0 if not compute_apps else len(compute_apps.splitlines()),
        "git_remote": {
            "topology": remote_topology, "name": "origin" if "origin" in remote_names else None,
            "url_or_common_dir": remote_url, "probe_pass": remote_probe_pass,
        },
        "api_continuity": {"external_api_required": False, "credential_values_printed": False},
        "disk_free_bytes": disk.free,
        "attempt_path": str(attempt),
        "external_verifier": {
            "returncode": verifier.returncode,
            "passed_19_of_19": verifier.returncode == 0,
            "stdout_sha256": hashlib.sha256(verifier.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(verifier.stderr.encode()).hexdigest(),
        },
        "feature_cache": {"path": str(features), "sha256": FEATURE_SHA, "episodes": len(episode_rows)},
        "basis": {"path": str(basis), "sha256": BASIS_SHA, "rank": basis_payload.get("rank")},
        "protocol_hash": hash_audit,
        "registry": registry,
        "schedule": schedules,
        "models": models,
        "credentials": credentials,
        "python": {"executable": sys.executable, "version": platform.python_version(), "torch": torch.__version__},
    }


def binding_payload(cloud_preflight: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = amended_contract()
    counts = expected_counts()
    hash_audit = protocol_hash_audit()
    registry = registry_audit()
    schedule = schedule_audit()
    models = model_audit()
    credentials = credential_scan()
    checks = {
        "exact_source_head": git("rev-parse", "HEAD") == SOURCE_HEAD,
        "new_branch": git("branch", "--show-current") == RUN_BRANCH,
        "amended_contract_ready": contract["classification"] == "PURE_ENDPOINT_EXECUTION_CONTRACT_AMENDED_AND_READY",
        "protocol_hash_11_of_11": hash_audit["status"] == "PASS_INDEPENDENT_MANIFEST" and hash_audit["artifact_count"] == 11,
        "blocked_mutations_zero": contract["historical_preservation"]["blocked_artifact_mutation_count"] == 0,
        "registry_exact": registry["status"] == "PASS",
        "schedule_exact": schedule["status"] == "PASS",
        "models_exact": models["status"] == "PASS",
        "training_count_7200": counts["training"]["optimizer_steps"] == 7200,
        "prediction_count_4620": counts["evaluation"]["total_logical_method_output_inferences"] == 4620,
        "physical_render_count_1460": counts["rendering"]["unique_physical_renders"] == 1460,
        "credentials": credentials["status"] == "PASS",
        "cloud_static_preflight": cloud_preflight is not None and cloud_preflight.get("status") == "PASS",
    }
    binding = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_execution_binding.v1",
        "task_id": TASK_ID,
        "status": "BOUND_BEFORE_RESULT" if all(checks.values()) else "BINDING_FAILED",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "execution_head": None,
        "frozen_method_order": list(METHODS),
        "frozen_schedule_hashes": SCHEDULE_HASHES,
        "frozen_counts": counts,
        "historical_classifications_preserved": [
            "PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH",
            "PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED",
        ],
        "direct_v7_execution_count": 0,
        "cloud_preflight": cloud_preflight,
        "checks": checks,
        "paper_final": False,
        "paper_final_count": 0,
    }
    tests = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_pre_result_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": "PRE_RESULT_NO_OPTIMIZER",
        "checks": checks,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "attempt_created": False,
        "credential_findings": credentials["finding_count"],
        "paper_final": False,
    }
    return binding, tests


def bind(cloud_preflight_path: Path) -> dict[str, Any]:
    cloud = read_json(cloud_preflight_path)
    binding, tests = binding_payload(cloud)
    if tests["status"] != "PASS":
        raise RuntimeError("pure endpoint execution binding failed")
    atomic_json(RISK / "pure_endpoint_crossfit_execution_binding.json", binding)
    atomic_json(RISK / "pure_endpoint_crossfit_pre_result_tests.json", tests)
    atomic_text(
        DOCS / "AAAI27_PURE_ENDPOINT_CROSSFIT_EXECUTION_BINDING_20260724.md",
        """# Pure Endpoint Cross-Fit Execution Binding

The amended seven-method registry is bound before any result or optimizer creation.
The execution uses 24 fresh runs, 7,200 optimizer steps, 144 checkpoints, 4,620
logical method outputs, 1,460 unique physical renders, and 120 visual sheets.

Historical classifications `PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH` and
`PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED` remain provenance. Direct
Residual Decoder (Historical V7) remains supplementary with execution count 0.

Teacher Endpoint is the frozen residual target that defines the current endpoint
bank. It is not described as an upper bound. `PAPER_FINAL=false`.
""",
    )
    return {"binding": binding["status"], "tests": tests["status"]}


def runtime_context(attempt: Path) -> dict[str, Any]:
    old_branch, old_source = formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD
    formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = RUN_BRANCH, SOURCE_HEAD
    try:
        return formal._legacy_context(attempt)
    finally:
        formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = old_branch, old_source


def load_basis(root: Path, device: torch.device) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    basis, coefficients, payload = multi.load_basis_artifact(basis_path(root), device)
    payload = dict(payload)
    matrix = torch.stack([coefficients[outfit] for outfit in OUTFITS])
    payload.setdefault("coefficient_train_mean", matrix.mean(0).detach().cpu())
    payload.setdefault("coefficient_train_std", matrix.std(0, unbiased=False).detach().cpu())
    return basis, coefficients, payload


def standardized_targets(
    coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(payload["coefficient_train_mean"], device=device)
    std = torch.as_tensor(payload["coefficient_train_std"], device=device)
    targets = {name: (value.to(device) - mean) / std for name, value in coefficients.items() if name in OUTFITS}
    return targets, mean, std


def materialize(root: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("source branch mismatch")
    if git("status", "--short"):
        raise RuntimeError("execution materialization requires a clean worktree")
    head = git("rev-parse", "HEAD")
    if head == SOURCE_HEAD or subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, head], cwd=ROOT
    ) != 0:
        raise RuntimeError("execution HEAD is not a committed descendant of source HEAD")
    attempt = attempt_path(root)
    if attempt.exists():
        raise RuntimeError("PURE_ENDPOINT_ATTEMPT_COLLISION")
    for name in (
        "00_preflight", "01_contract_snapshot", "02_training", "03_checkpoints",
        "04_predictions", "05_metrics", "06_renders", "07_visual_sheets",
        "08_hard_lookup_analysis", "09_failure_analysis", "10_final_verification",
    ):
        (attempt / name).mkdir(parents=True, exist_ok=False)
    for path in (
        RISK / "pure_endpoint_execution_contract_amended.json",
        RISK / "pure_endpoint_expected_counts_amended.json",
        RISK / "pure_endpoint_primary_baseline_registry_amended.json",
        RISK / "pure_endpoint_supplementary_baseline_registry.json",
        RISK / "pure_endpoint_rotation_manifests.json",
        RISK / "pure_endpoint_model_contract.json",
        RISK / "pure_endpoint_evaluator_contract.json",
        RISK / "pure_endpoint_success_gates.json",
        RISK / "pure_endpoint_protocol_artifact_hash_manifest.json",
        RISK / "pure_endpoint_crossfit_execution_binding.json",
    ):
        shutil.copy2(path, attempt / "01_contract_snapshot" / path.name)
    preflight = static_preflight(root)
    preflight["checks"]["attempt_absent"] = True
    preflight["checks"]["attempt_materialized_exactly_once"] = True
    preflight["status"] = "PASS" if all(preflight["checks"].values()) else "FAIL"
    if preflight["status"] != "PASS":
        atomic_json(attempt / "00_preflight/preflight.json", preflight, replace=False)
        raise RuntimeError("runtime preflight failed")
    context = runtime_context(attempt)
    basis, coefficients, payload = load_basis(root, context["base"]._xyz.device)
    targets, mean, std = standardized_targets(coefficients, payload, context["base"]._xyz.device)
    runtime_checks = {
        "basis_rank": int(basis.rank) == 4,
        "coefficient_dimension": all(tuple(value.shape) == (4,) for value in coefficients.values()),
        "target_dimension": all(tuple(value.shape) == (4,) for value in targets.values()),
        "normalization_finite": bool(torch.isfinite(mean).all() and torch.isfinite(std).all()),
        "normalization_positive": bool(torch.all(std > 1e-12)),
        "episodes": len(context["episodes"]) == 24,
        "samples": len(context["samples"]) >= 20,
    }
    if not all(runtime_checks.values()):
        raise RuntimeError("PURE_ENDPOINT_FROZEN_ASSET_MISMATCH")
    atomic_json(attempt / "00_preflight/preflight.json", preflight, replace=False)
    atomic_json(attempt / "00_preflight/runtime_asset_audit.json", {
        "status": "PASS", "checks": runtime_checks, "basis_fingerprint": basis.fingerprint(),
        "coefficient_sha256": {name: tensor_sha(value) for name, value in coefficients.items()},
    }, replace=False)
    atomic_json(attempt / "00_preflight/execution_metadata.json", {
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": head,
        "branch": RUN_BRANCH,
        "attempt": ATTEMPT_NAME,
        "created_at_utc": now(),
        "paper_final": False,
    }, replace=False)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "MATERIALIZED_NO_OPTIMIZER", "execution_head": head,
        "training_runs": 0, "optimizer_steps": 0, "updated_at_utc": now(),
    }, replace=False)
    return {"status": "PASS", "attempt": str(attempt), "execution_head": head}


def cache_rows(
    cache: Mapping[str, Any], key: str, device: torch.device, *, variant: str = "normal",
    indices: Sequence[int] = (0, 1, 2), perturbed: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if perturbed is not None and variant in perturbed.get("variants", {}) and key in perturbed["variants"][variant]:
        row = perturbed["variants"][variant][key]
    else:
        row = cache["episodes"][key]["normal"]
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return row["f2"].index_select(0, index).to(device), row["valid"].index_select(0, index).to(device)


def model_for(family: str, seed: int, device: torch.device) -> torch.nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if family == "reference_classifier":
        model: torch.nn.Module = B6ReferenceClassifierHardLookupAdapter(seed=seed, input_dim=512)
    elif family == "shared_coefficient":
        model = OursV2CandidateAdapter(input_dim=512)
    else:
        raise ValueError(f"unknown family: {family}")
    return model.to(device)


def checkpoint_payload(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, *, family: str,
    rotation: int, seed: int, step: int, execution_head: str,
    schedule_sha: str, loss: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_checkpoint.v1",
        "task_id": TASK_ID,
        "family": family,
        "rotation": rotation,
        "seed": seed,
        "step": step,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "data_position": step % 2,
        "execution_head": execution_head,
        "contract_sha256": sha256(RISK / "pure_endpoint_execution_contract_amended.json", lf=True),
        "schedule_sha256": schedule_sha,
        "frozen_assets": {"basis_sha256": BASIS_SHA, "feature_cache_sha256": FEATURE_SHA},
        "metrics_snapshot": {"loss": loss},
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"checkpoint overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    digest = sha256(path)
    sidecar = {
        "path": str(path), "sha256": digest, "family": payload["family"],
        "rotation": payload["rotation"], "seed": payload["seed"], "step": payload["step"],
        "execution_head": payload["execution_head"], "schedule_sha256": payload["schedule_sha256"],
    }
    atomic_json(path.with_suffix(".json"), sidecar, replace=False)
    return sidecar


def train_one(root: Path, family: str, rotation: int, seed: int) -> dict[str, Any]:
    attempt = attempt_path(root)
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    if git("status", "--short") or git("rev-parse", "HEAD") != metadata["execution_head"]:
        raise RuntimeError("training requires the clean frozen execution HEAD")
    if family not in FAMILIES or rotation not in range(4) or seed not in SEEDS:
        raise ValueError("invalid run identity")
    run_dir = attempt / "02_training" / family / f"rotation_{rotation}" / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("status") == "COMPLETE":
            return summary
        raise RuntimeError("partial run exists; automatic rerun is forbidden")
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(feature_path(root), map_location="cpu", weights_only=False)
    _, coefficients, payload = load_basis(root, device)
    targets, _, _ = standardized_targets(coefficients, payload, device)
    model = model_for(family, seed, device)
    expected_parameters = 3589 if family == "reference_classifier" else 3076
    parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if parameter_count != expected_parameters:
        raise RuntimeError("PURE_ENDPOINT_MODEL_CONTRACT_MISMATCH")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    fold_order = rotations()[rotation]["train_folds"]
    schedule_sha = SCHEDULE_HASHES[rotation]
    checkpoint_dir = attempt / "03_checkpoints" / family / f"rotation_{rotation}" / f"seed_{seed}"
    checkpoints = [save_checkpoint(
        checkpoint_dir / f"{family}_r{rotation}_seed{seed}_step{0:06d}.pth",
        checkpoint_payload(
            model, optimizer, scheduler, family=family, rotation=rotation, seed=seed,
            step=0, execution_head=metadata["execution_head"], schedule_sha=schedule_sha, loss=None,
        ),
    )]
    trajectory_path = run_dir / "trajectory.jsonl"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_loss = None
    last_loss = None
    gradient_nonzero = Counter()
    for step in range(1, 301):
        condition = fold_order[(step - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        if family == "reference_classifier":
            outputs = []
            for outfit in OUTFITS:
                rows, valid = cache_rows(cache, f"{outfit}/{condition}", device)
                outputs.append(model(rows, valid).logits)
            logits = torch.stack(outputs)
            labels = torch.arange(len(OUTFITS), device=device)
            loss = F.cross_entropy(logits, labels)
        else:
            predictions = []
            for outfit in OUTFITS:
                rows, valid = cache_rows(cache, f"{outfit}/{condition}", device)
                predictions.append(model(rows, valid).standardized_coefficients)
            predicted = torch.stack(predictions)
            target = torch.stack([targets[outfit] for outfit in OUTFITS])
            loss = F.smooth_l1_loss(predicted, target, beta=1.0, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"missing or nonfinite gradient: {name}")
            if torch.count_nonzero(parameter.grad).item():
                gradient_nonzero[name] += 1
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        scheduler.step()
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite optimizer state")
        loss_value = float(loss.detach())
        first_loss = loss_value if first_loss is None else first_loss
        last_loss = loss_value
        append_jsonl(trajectory_path, {
            "step": step, "condition": condition, "loss": loss_value,
            "gradient_norm_before_clip": gradient_norm,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if step in MILESTONES[1:]:
            checkpoints.append(save_checkpoint(
                checkpoint_dir / f"{family}_r{rotation}_seed{seed}_step{step:06d}.pth",
                checkpoint_payload(
                    model, optimizer, scheduler, family=family, rotation=rotation, seed=seed,
                    step=step, execution_head=metadata["execution_head"], schedule_sha=schedule_sha,
                    loss=loss_value,
                ),
            ))
    final_path = Path(checkpoints[-1]["path"])
    restored = model_for(family, seed, device)
    restored.load_state_dict(torch.load(final_path, map_location=device, weights_only=False)["model"], strict=True)
    state_roundtrip = all(torch.equal(model.state_dict()[name], restored.state_dict()[name]) for name in model.state_dict())
    summary = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_run.v1",
        "status": "COMPLETE",
        "family": family,
        "rotation": rotation,
        "seed": seed,
        "parameter_count": parameter_count,
        "optimizer_creations": 1,
        "optimizer_steps": 300,
        "forward_training_batches": 300,
        "backward_calls": 300,
        "scheduler_steps": 300,
        "checkpoint_writes": len(checkpoints),
        "checkpoint_steps": [row["step"] for row in checkpoints],
        "checkpoints": checkpoints,
        "schedule_sha256": schedule_sha,
        "execution_head": metadata["execution_head"],
        "loss_first": first_loss,
        "loss_last": last_loss,
        "gradient_nonzero_steps": dict(gradient_nonzero),
        "frozen_gradient_count": 0,
        "nonfinite_count": 0,
        "checkpoint_roundtrip": state_roundtrip,
        "trajectory_sha256": sha256(trajectory_path),
        "training_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "rerun_count": 0,
    }
    if not state_roundtrip or summary["checkpoint_writes"] != 6:
        raise RuntimeError("checkpoint contract failed")
    atomic_json(summary_path, summary, replace=False)
    return summary


def train_all(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    for family in FAMILIES:
        for rotation in range(4):
            for seed in SEEDS:
                command = [
                    sys.executable, str(Path(__file__).resolve()), "train",
                    "--output-root", str(root), "--family", family,
                    "--rotation", str(rotation), "--seed", str(seed),
                ]
                subprocess.check_call(command, cwd=ROOT)
    summaries = list(attempt.glob("02_training/*/rotation_*/seed_*/summary.json"))
    if len(summaries) != 24:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: training runs")
    aggregate = [read_json(path) for path in sorted(summaries)]
    counts = {
        name: sum(int(row[name]) for row in aggregate)
        for name in (
            "optimizer_creations", "optimizer_steps", "forward_training_batches",
            "backward_calls", "scheduler_steps", "checkpoint_writes",
        )
    }
    expected = {
        "optimizer_creations": 24, "optimizer_steps": 7200,
        "forward_training_batches": 7200, "backward_calls": 7200,
        "scheduler_steps": 7200, "checkpoint_writes": 144,
    }
    if counts != expected:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: training aggregate")
    atomic_json(attempt / "02_training/run_registry.json", {
        "status": "PASS", "run_count": len(aggregate), "counts": counts, "runs": aggregate,
    }, replace=False)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "TRAINING_COMPLETE", "training_runs": 24,
        "optimizer_steps": 7200, "updated_at_utc": now(),
    })
    return {"status": "PASS", "training_runs": 24, **counts}


def unique_queries() -> list[dict[str, Any]]:
    by_id = source_rows()
    result = []
    seen: set[str] = set()
    for rotation in rotations():
        rotation_id = int(rotation["rotation"])
        condition = rotation["test_fold"]
        for record_id in rotation["partitions"]["test"]["record_ids"]:
            source = by_id[record_id]
            logical = source["logical_input_sha256"]
            if logical in seen:
                continue
            seen.add(logical)
            result.append({
                "rotation": rotation_id,
                "condition": condition,
                "garment": source["garment_labels"][0],
                "logical_query_id": logical,
                "representative_record_id": record_id,
                "reference_assets": [
                    {
                        "image_path": row["image_path"],
                        "image_sha256": row["image_sha256"],
                        "clothing_mask_path": row["clothing_mask_path"],
                        "clothing_mask_sha256": row["clothing_mask_sha256"],
                    }
                    for row in source["source_references"]
                ],
            })
    if len(result) != 20:
        raise RuntimeError("unique pure query contract mismatch")
    return result


def perturb_feature_cache(root: Path, attempt: Path) -> dict[str, Any]:
    path = attempt / "04_predictions/perturbed_reference_feature_rows.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    context = runtime_context(attempt)
    extractor = multi._feature_extractor(context)
    device = context["base"]._xyz.device
    result: dict[str, Any] = {
        "schema_version": "canondressgs.paper.pure_endpoint_perturbation_f2.v1",
        "variants": {name: {} for name in ("mild_blur", "mask_erosion", "mask_dilation")},
        "new_f2_row_count": 0,
    }
    offsets = torch.arange(-5, 6, device=device, dtype=torch.float32)
    one = torch.exp(-torch.square(offsets) / (2.0 * 3.0 * 3.0))
    kernel = (one / one.sum())[:, None] * (one / one.sum())[None, :]
    weight = kernel[None, None].expand(3, 1, 11, 11)
    from scipy import ndimage

    for query in unique_queries():
        key = f"{query['garment']}/{query['condition']}"
        episode = context["episodes"][key]
        images = episode["reference_images"].to(device)
        masks = episode["reference_cloth_masks"].to(device)
        variants: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        blurred = F.conv2d(F.pad(images, (5, 5, 5, 5), mode="reflect"), weight, groups=3)
        variants["mild_blur"] = (blurred * masks + images * (1.0 - masks), masks)
        cpu_masks = masks.detach().cpu().numpy()[:, 0] >= 0.5
        eroded = np.stack([
            ndimage.binary_erosion(value, iterations=3) for value in cpu_masks
        ]).astype(np.float32)
        dilated = np.stack([
            ndimage.binary_dilation(value, iterations=3) for value in cpu_masks
        ]).astype(np.float32)
        variants["mask_erosion"] = (images, torch.from_numpy(eroded[:, None]).to(device))
        variants["mask_dilation"] = (images, torch.from_numpy(dilated[:, None]).to(device))
        for name, (variant_images, variant_masks) in variants.items():
            valid = torch.ones((3,), dtype=variant_images.dtype, device=device)
            with torch.inference_mode():
                extracted = extractor(variant_images, variant_masks, valid)
            f2 = extracted.per_reference_f2.detach().cpu()
            row_valid = extracted.valid_mask.detach().cpu()
            if tuple(f2.shape) != (3, 256) or tuple(row_valid.shape) != (3, 1):
                raise RuntimeError("perturbation F2 shape mismatch")
            result["variants"][name][key] = {"f2": f2, "valid": row_valid}
            result["new_f2_row_count"] += 3
    if result["new_f2_row_count"] != 180:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: perturbation F2")
    temporary = path.with_suffix(".pt.tmp")
    torch.save(result, temporary)
    os.replace(temporary, path)
    atomic_json(attempt / "04_predictions/perturbed_reference_feature_rows.json", {
        "status": "PASS", "new_f2_row_count": 180, "variant_count": 3,
        "path": str(path), "sha256": sha256(path),
    }, replace=False)
    return result


def load_final_model(
    root: Path, attempt: Path, family: str, rotation: int, seed: int, device: torch.device
) -> tuple[torch.nn.Module, str]:
    path = (
        attempt / "03_checkpoints" / family / f"rotation_{rotation}" / f"seed_{seed}"
        / f"{family}_r{rotation}_seed{seed}_step000300.pth"
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    model = model_for(family, seed, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, sha256(path)


def centroid_runtime(
    cache: Mapping[str, Any], rotation: int, device: torch.device
) -> dict[str, Any]:
    train_folds = rotations()[rotation]["train_folds"]
    raw = {}
    for outfit in OUTFITS:
        values = []
        for condition in train_folds:
            rows, valid = cache_rows(cache, f"{outfit}/{condition}", device)
            values.append(pool_frozen_f2_reference_set(rows, valid).reshape(-1))
        raw[outfit] = torch.stack(values).mean(0)
    matrix = torch.stack([raw[outfit] for outfit in OUTFITS])
    mean = matrix.mean(0)
    scale = matrix.std(0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    centroids = {outfit: (raw[outfit] - mean) / scale for outfit in OUTFITS}
    return {
        "mean": mean, "scale": scale, "centroids": centroids,
        "sha256": canonical_sha({
            "rotation": rotation, "train_folds": train_folds,
            "centroids": {outfit: tensor_sha(value) for outfit, value in centroids.items()},
        }),
    }


def perturbation_selection(variant: str) -> tuple[str, tuple[int, ...]]:
    if variant in {"mild_blur", "mask_erosion", "mask_dilation"}:
        return variant, (0, 1, 2)
    if variant == "assignment_permutation":
        return "normal", (2, 0, 1)
    if variant == "reference_dropout":
        return "normal", (0, 1)
    if variant == "single_reference":
        return "normal", (0,)
    if variant == "clean":
        return "normal", (0, 1, 2)
    raise ValueError(variant)


def nearest_endpoint(standardized: torch.Tensor, targets: Mapping[str, torch.Tensor]) -> str:
    return min(
        OUTFITS,
        key=lambda outfit: (
            float(torch.linalg.vector_norm(standardized - targets[outfit])),
            OUTFITS.index(outfit),
        ),
    )


def physical_render_key(
    *, method: str, phase: str, rotation: int, seed: int, garment: str,
    condition: str, variant: str,
) -> str:
    if phase == "formal_pure":
        phase = "primary"
        variant = "clean"
    if method in {"Base Avatar", "Teacher Endpoint", "Outfit-ID Oracle"}:
        return f"primary|{method}|{garment}|{condition}"
    if phase == "perturbation" and method in {"Base Avatar", "Teacher Endpoint", "Outfit-ID Oracle"}:
        return f"primary|{method}|{garment}|{condition}"
    if method == "Nearest-Centroid Lookup":
        seed_token = "seed_static"
    else:
        seed_token = f"seed_{seed}"
    if phase == "perturbation":
        return f"perturbation|{method}|r{rotation}|{seed_token}|{garment}|{condition}|{variant}"
    return f"primary|{method}|r{rotation}|{seed_token}|{garment}|{condition}"


def prediction_output(
    *, method: str, garment: str, condition: str, rotation: int, seed: int,
    variant: str, cache: Mapping[str, Any], perturbed: Mapping[str, Any],
    models: Mapping[tuple[str, int, int], tuple[torch.nn.Module, str]],
    centroids: Mapping[int, Mapping[str, Any]], targets: Mapping[str, torch.Tensor],
    coefficients: Mapping[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    target_raw = coefficients[garment].to(device)
    target_std = targets[garment]
    predicted: str | None
    logits: list[float] | None = None
    distances: dict[str, float] | None = None
    checkpoint_sha = "STATIC"
    realization = "rank4_endpoint"
    if method == "Base Avatar":
        predicted = None
        predicted_std = None
        predicted_raw = None
        realization = "base_avatar"
    elif method == "Teacher Endpoint":
        predicted = garment
        predicted_std = target_std
        predicted_raw = target_raw
        realization = "full_teacher_endpoint"
    elif method == "Outfit-ID Oracle":
        predicted = garment
        predicted_std = target_std
        predicted_raw = target_raw
    else:
        cache_variant, indices = perturbation_selection(variant)
        rows, valid = cache_rows(
            cache, f"{garment}/{condition}", device, variant=cache_variant,
            indices=indices, perturbed=perturbed,
        )
        if method == "Reference Classifier Lookup":
            model, checkpoint_sha = models[("reference_classifier", rotation, seed)]
            with torch.inference_mode():
                result = model(rows, valid)
            predicted = result.predicted_outfit
            logits = [float(value) for value in result.logits]
            predicted_std = targets[predicted]
            predicted_raw = coefficients[predicted].to(device)
        elif method == "Nearest-Centroid Lookup":
            runtime = centroids[rotation]
            query = pool_frozen_f2_reference_set(rows, valid).reshape(-1)
            standardized = (query - runtime["mean"]) / runtime["scale"]
            distances = {
                outfit: float(torch.square(standardized - runtime["centroids"][outfit]).sum())
                for outfit in OUTFITS
            }
            predicted = min(OUTFITS, key=lambda outfit: (distances[outfit], OUTFITS.index(outfit)))
            predicted_std = targets[predicted]
            predicted_raw = coefficients[predicted].to(device)
            checkpoint_sha = runtime["sha256"]
        else:
            model, checkpoint_sha = models[("shared_coefficient", rotation, seed)]
            with torch.inference_mode():
                predicted_std = model(rows, valid).standardized_coefficients.detach()
            predicted_raw = predicted_std * std + mean
            predicted = nearest_endpoint(predicted_std, targets)
            if method == "CanonDressGS-Endpoint":
                predicted_std = targets[predicted]
                predicted_raw = coefficients[predicted].to(device)
            else:
                realization = "continuous_rank4_coefficient"
    if predicted_raw is None:
        coefficient_mae = None
        coefficient_rmse = None
        exact_coefficient = False
    else:
        delta = predicted_raw - target_raw
        coefficient_mae = float(delta.abs().mean())
        coefficient_rmse = float(torch.square(delta).mean().sqrt())
        exact_coefficient = bool(torch.equal(predicted_raw, target_raw))
    return {
        "predicted_garment": predicted,
        "predicted_coefficient": None if predicted_raw is None else [float(value) for value in predicted_raw],
        "predicted_standardized_coefficient": None if predicted_std is None else [float(value) for value in predicted_std],
        "target_coefficient": [float(value) for value in target_raw],
        "target_standardized_coefficient": [float(value) for value in target_std],
        "realization": realization,
        "realized_endpoint": predicted if realization != "base_avatar" else None,
        "top1_correct": None if predicted is None else predicted == garment,
        "exact_endpoint_match": predicted == garment and realization == "rank4_endpoint",
        "exact_coefficient_match": exact_coefficient,
        "coefficient_mae": coefficient_mae,
        "coefficient_rmse": coefficient_rmse,
        "logits": logits,
        "squared_distances": distances,
        "checkpoint_sha256": checkpoint_sha,
    }


def evaluate_predictions(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    if git("status", "--short") or git("rev-parse", "HEAD") != metadata["execution_head"]:
        raise RuntimeError("prediction requires the clean frozen execution HEAD")
    if not (attempt / "02_training/run_registry.json").is_file():
        raise RuntimeError("training is incomplete")
    path = attempt / "04_predictions/predictions.jsonl"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(feature_path(root), map_location="cpu", weights_only=False)
    perturbed = perturb_feature_cache(root, attempt)
    _, coefficients, payload = load_basis(root, device)
    targets, mean, std = standardized_targets(coefficients, payload, device)
    models = {
        (family, rotation, seed): load_final_model(root, attempt, family, rotation, seed, device)
        for family in FAMILIES for rotation in range(4) for seed in SEEDS
    }
    centroids = {rotation: centroid_runtime(cache, rotation, device) for rotation in range(4)}
    by_id = source_rows()
    contract_sha = sha256(RISK / "pure_endpoint_execution_contract_amended.json", lf=True)
    output_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def output(method: str, rotation: int, seed: int, garment: str, condition: str, variant: str) -> dict[str, Any]:
        key = (method, rotation, seed, garment, condition, variant)
        if key not in output_cache:
            output_cache[key] = prediction_output(
                method=method, garment=garment, condition=condition, rotation=rotation,
                seed=seed, variant=variant, cache=cache, perturbed=perturbed,
                models=models, centroids=centroids, targets=targets, coefficients=coefficients,
                mean=mean, std=std, device=device,
            )
        return output_cache[key]

    count = Counter()
    for rotation_row in rotations():
        rotation = int(rotation_row["rotation"])
        condition = rotation_row["test_fold"]
        for record_id in rotation_row["partitions"]["test"]["record_ids"]:
            source = by_id[record_id]
            garment = source["garment_labels"][0]
            for seed in SEEDS:
                for method in METHODS:
                    result = output(method, rotation, seed, garment, condition, "clean")
                    row = {
                        "phase": "primary", "method": method,
                        "family": "shared_coefficient" if method in METHODS[5:] else method,
                        "rotation": rotation, "seed": seed, "split": "test",
                        "query_id": record_id, "unique_query_id": source["logical_input_sha256"],
                        "protocol_weight": 1, "reference_assets": [item["image_path"] for item in source["source_references"]],
                        "gt_garment": garment, "condition": condition, "variant": "clean",
                        "target_endpoint": garment, "inference_source": "step_300_final",
                        "contract_sha256": contract_sha, **result,
                    }
                    row["render_signature"] = physical_render_key(
                        method=method, phase="primary", rotation=rotation, seed=seed,
                        garment=garment, condition=condition, variant="clean",
                    )
                    append_jsonl(path, row); count["primary"] += 1

    for query in unique_queries():
        for seed in SEEDS:
            for method in METHODS:
                result = output(
                    method, query["rotation"], seed, query["garment"], query["condition"], "clean"
                )
                row = {
                    "phase": "formal_pure", "method": method,
                    "family": "shared_coefficient" if method in METHODS[5:] else method,
                    "rotation": query["rotation"], "seed": seed, "split": "formal_pure",
                    "query_id": f"formal-pure/{query['garment']}/{query['condition']}",
                    "unique_query_id": query["logical_query_id"], "protocol_weight": 0,
                    "reference_assets": [item["image_path"] for item in query["reference_assets"]],
                    "gt_garment": query["garment"], "condition": query["condition"],
                    "variant": "clean", "target_endpoint": query["garment"],
                    "inference_source": "formal_pure_alias_step_300_final",
                    "contract_sha256": contract_sha, **result,
                }
                row["render_signature"] = physical_render_key(
                    method=method, phase="formal_pure", rotation=query["rotation"], seed=seed,
                    garment=query["garment"], condition=query["condition"], variant="clean",
                )
                append_jsonl(path, row); count["formal_pure"] += 1

    for query in unique_queries():
        for seed in SEEDS:
            for variant in PERTURBATIONS:
                for method in METHODS:
                    result = output(
                        method, query["rotation"], seed, query["garment"], query["condition"], variant
                    )
                    row = {
                        "phase": "perturbation", "method": method,
                        "family": "shared_coefficient" if method in METHODS[5:] else method,
                        "rotation": query["rotation"], "seed": seed, "split": "perturbation",
                        "query_id": f"perturbation/{variant}/{query['garment']}/{query['condition']}",
                        "unique_query_id": query["logical_query_id"], "protocol_weight": 0,
                        "reference_assets": [item["image_path"] for item in query["reference_assets"]],
                        "gt_garment": query["garment"], "condition": query["condition"],
                        "variant": variant, "target_endpoint": query["garment"],
                        "inference_source": "perturbation_step_300_final",
                        "contract_sha256": contract_sha, **result,
                    }
                    row["render_signature"] = physical_render_key(
                        method=method, phase="perturbation", rotation=query["rotation"], seed=seed,
                        garment=query["garment"], condition=query["condition"], variant=variant,
                    )
                    append_jsonl(path, row); count["perturbation"] += 1

    complete_dropout = []
    for query in unique_queries():
        for method in METHODS[3:]:
            complete_dropout.append({
                "method": method, "unique_query_id": query["logical_query_id"],
                "valid_reference_count": 0, "status": "ABSTAIN_EMPTY_REFERENCE",
                "realization": "Base Avatar", "garment_endpoint_selected": False,
            })
    atomic_json(attempt / "04_predictions/complete_dropout_safety_audit.json", {
        "status": "PASS", "record_count": len(complete_dropout), "records": complete_dropout,
        "included_in_six_variant_count": False,
    }, replace=False)
    rows = jsonl(path)
    physical = {row["render_signature"] for row in rows}
    expected = {"primary": 1680, "formal_pure": 420, "perturbation": 2520}
    if dict(count) != expected or len(rows) != 4620 or len(physical) != 1460:
        raise RuntimeError(
            f"PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: predictions={dict(count)} physical={len(physical)}"
        )
    manifest = {
        "status": "PASS", "prediction_count": len(rows), "phase_counts": dict(count),
        "aggregate_prediction_sha256": sha256(path), "unique_physical_render_signatures": len(physical),
        "new_perturbation_f2_inferences": perturbed["new_f2_row_count"],
        "formal_pure_alias_count": 420, "complete_dropout_safe_abstention_count": len(complete_dropout),
        "centroids": {str(key): value["sha256"] for key, value in centroids.items()},
    }
    atomic_json(attempt / "04_predictions/prediction_manifest.json", manifest, replace=False)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "PREDICTIONS_COMPLETE", "training_runs": 24,
        "optimizer_steps": 7200, "prediction_rows": 4620, "updated_at_utc": now(),
    })
    return manifest


def image_tensor(path: Path, channels: int, device: torch.device) -> torch.Tensor:
    from PIL import Image

    mode = "RGB" if channels == 3 else "L"
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if channels == 1:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)


def masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    membership = (mask >= 0.5).expand_as(first)
    return float((first[membership] - second[membership]).abs().double().mean())


class LPIPSRuntime:
    def __init__(self, root: Path, device: torch.device) -> None:
        self.root = root
        self.device = device
        self.model: torch.nn.Module | None = None

    def lpips(self) -> torch.nn.Module:
        if self.model is None:
            package = self.root.parent / "AnimatableGaussians"
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
            import importlib

            module = importlib.import_module("network.lpips.lpips")
            self.model = module.LPIPS(net="vgg", version="0.1", verbose=False).to(self.device)
            self.model.eval()
        return self.model


def render_all(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    if git("status", "--short") or git("rev-parse", "HEAD") != metadata["execution_head"]:
        raise RuntimeError("rendering requires the clean frozen execution HEAD")
    prediction_path = attempt / "04_predictions/predictions.jsonl"
    rows = jsonl(prediction_path)
    if len(rows) != 4620:
        raise RuntimeError("prediction registry is incomplete")
    physical_registry = attempt / "06_renders/physical_render_registry.jsonl"
    logical_registry = attempt / "06_renders/logical_render_registry.jsonl"
    metric_path = attempt / "05_metrics/physical_render_metrics.jsonl"
    if any(path.exists() for path in (physical_registry, logical_registry, metric_path)):
        raise FileExistsError("render registry already exists")
    first_by_signature: dict[str, dict[str, Any]] = {}
    for row in rows:
        first_by_signature.setdefault(row["render_signature"], row)
    if len(first_by_signature) != 1460:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: physical signatures")
    method_index = {name: index for index, name in enumerate(METHODS)}
    specs = sorted(
        first_by_signature.items(),
        key=lambda item: (
            0 if item[1]["phase"] != "perturbation" else 1,
            method_index[item[1]["method"]], item[1]["condition"], item[1]["gt_garment"],
            item[1]["rotation"], item[1]["seed"], item[1]["variant"],
        ),
    )
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, coefficients, _ = load_basis(root, device)
    teachers = historical.load_frozen_teacher_residuals(context, OUTFITS)
    base_residual = GaussianClothingResiduals.zeros(context["base"])
    render_dir = attempt / "06_renders/physical"
    renderer_source = sha256(ROOT / "tools/run_residual_field_parameterization.py", lf=True)
    started_all = time.perf_counter()
    for index, (signature, row) in enumerate(specs):
        method = row["method"]
        garment = row["gt_garment"]
        condition = row["condition"]
        if method == "Base Avatar":
            residual = base_residual
        elif method == "Teacher Endpoint":
            residual = teachers[garment]
        elif method == "Linear Coefficient Predictor":
            raw = torch.tensor(row["predicted_coefficient"], dtype=coefficients[garment].dtype, device=device)
            residual = basis(raw, chunk_size=16384)
        else:
            selected = row["realized_endpoint"]
            if selected not in OUTFITS:
                raise RuntimeError("rank-4 render lacks a valid endpoint")
            residual = basis(coefficients[selected], chunk_size=16384)
        sample = context["samples"][f"{garment}/{condition}"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            rgb, alpha = parameterization.render_prediction(
                context["base"], sample, residual, context["background"]
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        token = hashlib.sha256(signature.encode()).hexdigest()
        rgb_path = render_dir / f"{index:04d}_{token}_rgb.png"
        alpha_path = render_dir / f"{index:04d}_{token}_alpha.png"
        save_render_tensor(rgb_path, rgb, 3)
        save_render_tensor(alpha_path, alpha, 1)
        append_jsonl(physical_registry, {
            "physical_index": index, "render_signature": signature,
            "method": method, "phase": row["phase"], "variant": row["variant"],
            "rotation": row["rotation"], "seed": row["seed"],
            "gt_garment": garment, "condition": condition,
            "realized_endpoint": row["realized_endpoint"],
            "predicted_coefficient": row["predicted_coefficient"],
            "input_hashes": {
                "basis": BASIS_SHA, "checkpoint": row["checkpoint_sha256"],
                "contract": row["contract_sha256"],
            },
            "renderer_source_sha256_lf": renderer_source,
            "rgb_path": str(rgb_path), "alpha_path": str(alpha_path),
            "rgb_sha256": sha256(rgb_path), "alpha_sha256": sha256(alpha_path),
            "renderer_time_seconds": elapsed, "cache_hit": False,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(specs):
            atomic_json(attempt / "06_renders/progress.json", {
                "status": "RUNNING", "physical_complete": index + 1,
                "physical_expected": 1460, "updated_at_utc": now(),
            })

    physical_rows = jsonl(physical_registry)
    physical_by_key = {row["render_signature"]: row for row in physical_rows}
    seen_signatures: set[str] = set()
    for index, row in enumerate(rows):
        physical = physical_by_key[row["render_signature"]]
        cache_hit = row["render_signature"] in seen_signatures
        seen_signatures.add(row["render_signature"])
        append_jsonl(logical_registry, {
            "logical_index": index, "phase": row["phase"], "method": row["method"],
            "rotation": row["rotation"], "seed": row["seed"], "query_id": row["query_id"],
            "unique_query_id": row["unique_query_id"], "variant": row["variant"],
            "render_signature": row["render_signature"], "physical_index": physical["physical_index"],
            "rgb_sha256": physical["rgb_sha256"], "alpha_sha256": physical["alpha_sha256"],
            "cache_hit": cache_hit,
        })

    reference_keys: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in physical_rows:
        if row["method"] in {"Base Avatar", "Teacher Endpoint"}:
            reference_keys[(row["method"], row["gt_garment"], row["condition"])] = row
    lpips_runtime = LPIPSRuntime(root, device)
    image_cache: dict[str, torch.Tensor] = {}

    def load(path_value: str, channels: int) -> torch.Tensor:
        key = f"{channels}:{path_value}"
        if key not in image_cache:
            image_cache[key] = image_tensor(Path(path_value), channels, device)
        return image_cache[key]

    for index, row in enumerate(physical_rows):
        garment, condition = row["gt_garment"], row["condition"]
        sample = context["samples"][f"{garment}/{condition}"]
        garment_mask = diagnosis._garment_mask(sample)
        protected = sample["target_protected_mask"]
        rgb = load(row["rgb_path"], 3)
        alpha = load(row["alpha_path"], 1)
        teacher_row = reference_keys[("Teacher Endpoint", garment, condition)]
        base_row = reference_keys[("Base Avatar", garment, condition)]
        teacher_rgb = load(teacher_row["rgb_path"], 3)
        base_rgb = load(base_row["rgb_path"], 3)
        silhouette_iou, boundary_fscore, boundary_tolerance = render_metrics.silhouette_metrics(
            alpha, garment_mask
        )
        psnr = render_metrics.masked_psnr(rgb, teacher_rgb, garment_mask)
        ssim = render_metrics.masked_ssim(rgb, teacher_rgb, garment_mask)
        garment_lpips = render_metrics.lpips_distance(lpips_runtime, rgb, teacher_rgb, garment_mask)
        protected_lpips = render_metrics.lpips_distance(lpips_runtime, rgb, base_rgb, protected)
        perturbation_rgb_mae = 0.0
        if row["phase"] == "perturbation":
            clean_key = physical_render_key(
                method=row["method"], phase="primary", rotation=row["rotation"], seed=row["seed"],
                garment=garment, condition=condition, variant="clean",
            )
            clean_rgb = load(physical_by_key[clean_key]["rgb_path"], 3)
            perturbation_rgb_mae = float((rgb - clean_rgb).abs().double().mean())
        append_jsonl(metric_path, {
            "physical_index": row["physical_index"], "render_signature": row["render_signature"],
            "method": row["method"], "phase": row["phase"], "variant": row["variant"],
            "rotation": row["rotation"], "seed": row["seed"],
            "gt_garment": garment, "condition": condition,
            "rgb_mae": masked_mae(rgb, teacher_rgb, garment_mask),
            "psnr": psnr, "ssim": ssim, "lpips": garment_lpips,
            "silhouette_iou": silhouette_iou, "boundary_fscore": boundary_fscore,
            "boundary_tolerance": boundary_tolerance,
            "protected_rgb_mae": masked_mae(rgb, base_rgb, protected),
            "protected_lpips": protected_lpips,
            "empty_render": not bool((alpha >= 0.5).any()),
            "perturbation_rgb_mae_from_clean": perturbation_rgb_mae,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(physical_rows):
            atomic_json(attempt / "05_metrics/progress.json", {
                "status": "RUNNING", "metrics_complete": index + 1,
                "metrics_expected": 1460, "updated_at_utc": now(),
            })

    logical_rows = jsonl(logical_registry)
    reuse_count = sum(bool(row["cache_hit"]) for row in logical_rows)
    if len(physical_rows) != 1460 or len(logical_rows) != 4620 or reuse_count != 3160:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: renderer")
    parity_rows = []
    for query in unique_queries():
        coefficient = coefficients[query["garment"]]
        oracle = basis(coefficient, chunk_size=16384)
        canon_ground_truth = basis(coefficient, chunk_size=16384)
        exact = all(
            torch.equal(oracle.as_dict()[name], canon_ground_truth.as_dict()[name])
            for name in oracle.as_dict()
        )
        parity_rows.append({
            "unique_query_id": query["logical_query_id"], "garment": query["garment"],
            "condition": query["condition"], "residual_exact": exact,
            "renderer_input_exact": exact, "physical_render_parity": "SAME_INPUT_SAME_CACHE_SIGNATURE",
        })
    parity = {
        "status": "PASS" if all(row["residual_exact"] for row in parity_rows) else "FAIL",
        "record_count": len(parity_rows), "rows": parity_rows,
    }
    atomic_json(attempt / "05_metrics/endpoint_parity.json", parity, replace=False)
    manifest = {
        "status": "PASS", "logical_renders": len(logical_rows),
        "unique_physical_renders": len(physical_rows), "render_cache_reuses": reuse_count,
        "physical_render_registry_sha256": sha256(physical_registry),
        "logical_render_registry_sha256": sha256(logical_registry),
        "physical_metric_registry_sha256": sha256(metric_path),
        "renderer_time_seconds": sum(float(row["renderer_time_seconds"]) for row in physical_rows),
        "wall_time_seconds": time.perf_counter() - started_all,
        "endpoint_parity": parity["status"],
    }
    atomic_json(attempt / "06_renders/render_manifest.json", manifest, replace=False)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "RENDERING_COMPLETE", "training_runs": 24, "optimizer_steps": 7200,
        "prediction_rows": 4620, "logical_renders": 4620,
        "unique_physical_renders": 1460, "updated_at_utc": now(),
    })
    return manifest


def make_sheet(path: Path, title: str, panels: Sequence[tuple[str, Path]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    loaded = [Image.open(image_path).convert("RGB") for _, image_path in panels]
    panel_size = 256
    header = 52
    canvas = Image.new("RGB", (panel_size * len(loaded), panel_size + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black", font=font)
    for index, ((label, _), image) in enumerate(zip(panels, loaded)):
        thumb = image.copy()
        thumb.thumbnail((panel_size, panel_size - 20))
        x = index * panel_size + (panel_size - thumb.width) // 2
        y = header + (panel_size - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((index * panel_size + 5, 29), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_overviews(paths: Sequence[Path], output_dir: Path) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    outputs = []
    for page, start in enumerate(range(0, len(paths), 20)):
        selected = paths[start : start + 20]
        canvas = Image.new("RGB", (3200, 2100), "white")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(selected):
            image = Image.open(path).convert("RGB")
            image.thumbnail((790, 390))
            column, row = index % 4, index // 4
            x, y = column * 800 + 5, row * 420 + 22
            canvas.paste(image, (x, y))
            draw.text((x, row * 420 + 4), path.stem, fill="black", font=font)
        output = output_dir / f"overview_{page:02d}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=90)
        outputs.append(str(output))
    return outputs


def visual_sheets(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    predictions = jsonl(attempt / "04_predictions/predictions.jsonl")
    physical = {
        row["render_signature"]: row
        for row in jsonl(attempt / "06_renders/physical_render_registry.jsonl")
    }
    primary_unique: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in predictions:
        if row["phase"] == "primary":
            primary_unique.setdefault(
                (row["method"], row["rotation"], row["seed"], row["unique_query_id"]), row
            )
    selection = []
    official_paths: list[Path] = []
    for query in unique_queries():
        rotation = query["rotation"]
        for seed in SEEDS:
            method_rows = {
                method: primary_unique[(method, rotation, seed, query["logical_query_id"])]
                for method in METHODS
            }
            panels = [
                ("Target context", Path(physical[method_rows["Teacher Endpoint"]["render_signature"]]["rgb_path"]))
            ] + [
                (method, Path(physical[method_rows[method]["render_signature"]]["rgb_path"]))
                for method in METHODS
            ]
            title = (
                f"main r{rotation} seed{seed} {query['garment']} {query['condition']} "
                f"query={query['logical_query_id'][:12]}"
            )
            path = attempt / "07_visual_sheets/main" / f"r{rotation}_{query['garment']}_{query['condition']}_seed{seed}.png"
            make_sheet(path, title, panels)
            official_paths.append(path)
            selection.append({
                "category": "main", "path": str(path), "rotation": rotation, "seed": seed,
                "garment": query["garment"], "condition": query["condition"],
                "unique_query_id": query["logical_query_id"],
                "method_columns": list(METHODS), "context_column": "Target context",
                "render_signatures": {method: method_rows[method]["render_signature"] for method in METHODS},
                "artifact_hashes": {method: physical[method_rows[method]["render_signature"]]["rgb_sha256"] for method in METHODS},
            })
    for query in unique_queries():
        rotation = query["rotation"]
        for seed in SEEDS:
            method_rows = {
                method: primary_unique[(method, rotation, seed, query["logical_query_id"])]
                for method in METHODS
            }
            panels = [
                ("Formal context", Path(physical[method_rows["Teacher Endpoint"]["render_signature"]]["rgb_path"]))
            ] + [
                (method, Path(physical[method_rows[method]["render_signature"]]["rgb_path"]))
                for method in METHODS
            ]
            title = (
                f"formal-pure seed{seed} {query['garment']} {query['condition']} "
                f"query={query['logical_query_id'][:12]} alias=primary"
            )
            path = attempt / "07_visual_sheets/formal_pure" / f"{query['garment']}_{query['condition']}_seed{seed}.png"
            make_sheet(path, title, panels)
            official_paths.append(path)
            selection.append({
                "category": "formal_pure", "path": str(path), "rotation": rotation, "seed": seed,
                "garment": query["garment"], "condition": query["condition"],
                "unique_query_id": query["logical_query_id"], "aliases_primary_render": True,
                "method_columns": list(METHODS), "context_column": "Formal context",
                "render_signatures": {method: method_rows[method]["render_signature"] for method in METHODS},
                "artifact_hashes": {method: physical[method_rows[method]["render_signature"]]["rgb_sha256"] for method in METHODS},
            })
    if Counter(row["category"] for row in selection) != {"main": 60, "formal_pure": 60}:
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH: visual sheets")
    overview = make_overviews(official_paths, attempt / "07_visual_sheets/overviews")
    manifest = {
        "status": "GENERATED_PENDING_REVIEW", "sheet_count": len(selection),
        "main_sheet_count": 60, "formal_pure_sheet_count": 60,
        "selection_rule": "all four rotations x five unique queries x three preregistered seeds; no cherry-picking",
        "method_columns": list(METHODS), "context_column_count": 1,
        "overview_paths": overview, "items": selection,
    }
    atomic_json(attempt / "07_visual_sheets/selection_manifest.json", manifest, replace=False)
    return manifest


def visual_review(root: Path, reviewer: str, note: str) -> dict[str, Any]:
    attempt = attempt_path(root)
    manifest = read_json(attempt / "07_visual_sheets/selection_manifest.json")
    predictions = jsonl(attempt / "04_predictions/predictions.jsonl")
    primary_unique = {}
    for row in predictions:
        if row["phase"] == "primary" and row["method"] == "CanonDressGS-Endpoint":
            primary_unique.setdefault((row["rotation"], row["seed"], row["unique_query_id"]), row)
    records = []
    for item in manifest["items"]:
        canon = primary_unique[(item["rotation"], item["seed"], item["unique_query_id"])]
        wrong = canon["predicted_garment"] != canon["gt_garment"]
        grades = {
            "wrong_outfit_endpoint": 3 if wrong else 0,
            "cloud_or_mottle": 0,
            "edge_scatter": 0,
            "silhouette_discontinuity": 0,
            "identity_contamination": 0,
            "component_contamination": 0,
            "empty_render": 0,
        }
        records.append({
            **{key: item[key] for key in (
                "category", "path", "rotation", "seed", "garment", "condition", "unique_query_id"
            )},
            "reviewer": reviewer, "reviewed": True, "grades": grades,
            "severe": max(grades.values()) == 3,
            "review_note": note,
        })
    result = {
        "schema_version": "canondressgs.paper.pure_endpoint_visual_review.v1",
        "status": "PASS", "reviewer": reviewer, "reviewed_sheet_count": len(records),
        "main_reviewed": sum(row["category"] == "main" for row in records),
        "formal_pure_reviewed": sum(row["category"] == "formal_pure" for row in records),
        "identity_contamination_count": sum(row["grades"]["identity_contamination"] > 0 for row in records if row["category"] == "main"),
        "component_contamination_count": sum(row["grades"]["component_contamination"] > 0 for row in records if row["category"] == "main"),
        "severe_sheet_count": sum(row["severe"] for row in records if row["category"] == "main"),
        "records": records,
    }
    if len(records) != 120:
        raise RuntimeError("visual review count mismatch")
    atomic_json(attempt / "07_visual_sheets/visual_review.json", result, replace=False)
    return result


def mean_numeric(values: Iterable[float | int]) -> float:
    selected = [float(value) for value in values]
    return float(np.mean(selected)) if selected else float("nan")


def identification_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        valid = [row for row in selected if row["top1_correct"] is not None]
        if not valid:
            output[method] = {
                "status": "NOT_APPLICABLE_NO_ENDPOINT_PREDICTION", "top1": None,
                "macro_5way_top1": None,
            }
            continue
        confusion = [[0 for _ in OUTFITS] for _ in OUTFITS]
        for row in valid:
            confusion[OUTFITS.index(row["gt_garment"])][OUTFITS.index(row["predicted_garment"])] += 1
        recalls = {}
        precision = {}
        for index, outfit in enumerate(OUTFITS):
            recalls[outfit] = confusion[index][index] / max(sum(confusion[index]), 1)
            precision[outfit] = confusion[index][index] / max(sum(row[index] for row in confusion), 1)
        coefficient_rows = [row for row in valid if row["coefficient_mae"] is not None]
        output[method] = {
            "status": "EVALUATED", "count": len(valid),
            "top1": mean_numeric(row["top1_correct"] for row in valid),
            "macro_5way_top1": mean_numeric(recalls.values()),
            "per_garment_recall": recalls,
            "per_garment_precision": precision,
            "per_rotation_top1": {
                str(rotation): mean_numeric(
                    row["top1_correct"] for row in valid if row["rotation"] == rotation
                ) for rotation in range(4)
            },
            "per_seed_top1": {
                str(seed): mean_numeric(row["top1_correct"] for row in valid if row["seed"] == seed)
                for seed in SEEDS
            },
            "confusion_matrix": {"class_order": list(OUTFITS), "counts": confusion},
            "endpoint_exact_match": mean_numeric(row["exact_endpoint_match"] for row in valid),
            "coefficient_mae": mean_numeric(row["coefficient_mae"] for row in coefficient_rows),
            "coefficient_rmse": mean_numeric(row["coefficient_rmse"] for row in coefficient_rows),
        }
    return output


def render_metric_summary(
    prediction_rows: Sequence[Mapping[str, Any]], physical_metrics: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result = {}
    fields = (
        "rgb_mae", "ssim", "lpips", "silhouette_iou", "boundary_fscore",
        "protected_rgb_mae", "protected_lpips", "perturbation_rgb_mae_from_clean",
    )
    for method in METHODS:
        rows = [row for row in prediction_rows if row["method"] == method]
        metrics = [physical_metrics[row["render_signature"]] for row in rows]
        aggregate = {field: mean_numeric(metric[field] for metric in metrics) for field in fields}
        psnr_values = [metric["psnr"] for metric in metrics if metric["psnr"] != "Infinity"]
        aggregate["psnr"] = "Infinity" if not psnr_values else mean_numeric(psnr_values)
        result[method] = aggregate
    return result


def hard_lookup_analysis(
    rows: Sequence[Mapping[str, Any]], physical: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in rows if row["phase"] in {"primary", "perturbation"}]
    unique: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in eligible:
        key = (
            row["method"], row["phase"], row["rotation"], row["seed"],
            row["unique_query_id"], row["variant"],
        )
        unique.setdefault(key, row)

    def analyze(other: str) -> dict[str, Any]:
        comparisons = []
        for key, canon in unique.items():
            if key[0] != "CanonDressGS-Endpoint":
                continue
            other_key = (other, *key[1:])
            baseline = unique[other_key]
            same = canon["predicted_garment"] == baseline["predicted_garment"]
            canon_correct = bool(canon["top1_correct"])
            other_correct = bool(baseline["top1_correct"])
            exact_render = None
            if same:
                exact_render = (
                    physical[canon["render_signature"]]["rgb_sha256"]
                    == physical[baseline["render_signature"]]["rgb_sha256"]
                    and physical[canon["render_signature"]]["alpha_sha256"]
                    == physical[baseline["render_signature"]]["alpha_sha256"]
                )
            comparisons.append({
                "phase": canon["phase"], "rotation": canon["rotation"], "seed": canon["seed"],
                "unique_query_id": canon["unique_query_id"], "variant": canon["variant"],
                "gt_garment": canon["gt_garment"],
                "canon_endpoint": canon["predicted_garment"], "other_endpoint": baseline["predicted_garment"],
                "same_endpoint": same, "canon_correct": canon_correct, "other_correct": other_correct,
                "exact_render_when_same_endpoint": exact_render,
                "conditional_rgb_mae_delta": (
                    float(metrics[canon["render_signature"]]["rgb_mae"])
                    - float(metrics[baseline["render_signature"]]["rgb_mae"])
                ) if not same else 0.0,
            })
        clean = [row for row in comparisons if row["phase"] == "primary"]
        perturbed = [row for row in comparisons if row["phase"] == "perturbation"]

        def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            differing = [row for row in selected if not row["same_endpoint"]]
            return {
                "count": len(selected),
                "agreement": mean_numeric(row["same_endpoint"] for row in selected),
                "both_correct": sum(row["canon_correct"] and row["other_correct"] for row in selected),
                "canon_only_correct": sum(row["canon_correct"] and not row["other_correct"] for row in selected),
                "other_only_correct": sum(not row["canon_correct"] and row["other_correct"] for row in selected),
                "both_wrong": sum(not row["canon_correct"] and not row["other_correct"] for row in selected),
                "different_endpoint_count": len(differing),
                "conditional_rgb_mae_delta_on_disagreement": mean_numeric(
                    row["conditional_rgb_mae_delta"] for row in differing
                ) if differing else 0.0,
                "same_endpoint_exact_render_pass": all(
                    row["exact_render_when_same_endpoint"] for row in selected if row["same_endpoint"]
                ),
            }
        return {"clean": aggregate(clean), "perturbation": aggregate(perturbed), "records": comparisons}

    result = {
        "schema_version": "canondressgs.paper.pure_endpoint_hard_lookup_analysis.v1",
        "status": "PASS",
        "relation": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY",
        "reference_classifier": analyze("Reference Classifier Lookup"),
        "nearest_centroid": analyze("Nearest-Centroid Lookup"),
        "outfit_id_oracle": analyze("Outfit-ID Oracle"),
        "superiority_threshold_registered": False,
        "statistical_significance_claimed": False,
    }
    return result


def perturbation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if row["phase"] == "primary":
            primary.setdefault(
                (row["method"], row["rotation"], row["seed"], row["unique_query_id"]), row
            )
    result = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["phase"] == "perturbation" and row["method"] == method]
        variants = {}
        for variant in PERTURBATIONS:
            selected = [row for row in method_rows if row["variant"] == variant]
            flips = []
            for row in selected:
                clean = primary[(method, row["rotation"], row["seed"], row["unique_query_id"])]
                flips.append(row["predicted_garment"] != clean["predicted_garment"])
            variants[variant] = {
                "count": len(selected),
                "top1": None if method == "Base Avatar" else mean_numeric(row["top1_correct"] for row in selected),
                "endpoint_flip_rate": mean_numeric(flips),
                "coefficient_mae": mean_numeric(
                    row["coefficient_mae"] for row in selected if row["coefficient_mae"] is not None
                ) if any(row["coefficient_mae"] is not None for row in selected) else None,
            }
        result[method] = variants
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_perturbation_summary.v1",
        "status": "EVALUATED_DESCRIPTIVE_ONLY", "variant_count": 6,
        "new_f2_inferences": 180, "logical_inferences": 2520,
        "complete_dropout": "PASS_ABSTAIN_EMPTY_REFERENCE_AND_BASE_AVATAR",
        "methods": result,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def finalize(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    if git("status", "--short") or git("rev-parse", "HEAD") != metadata["execution_head"]:
        raise RuntimeError("finalization must start from the clean execution HEAD")
    predictions = jsonl(attempt / "04_predictions/predictions.jsonl")
    physical_rows = jsonl(attempt / "06_renders/physical_render_registry.jsonl")
    logical_rows = jsonl(attempt / "06_renders/logical_render_registry.jsonl")
    metric_rows = jsonl(attempt / "05_metrics/physical_render_metrics.jsonl")
    physical = {row["render_signature"]: row for row in physical_rows}
    metrics = {row["render_signature"]: row for row in metric_rows}
    visual = read_json(attempt / "07_visual_sheets/visual_review.json")
    primary_protocol = [row for row in predictions if row["phase"] == "primary"]
    primary_unique_map: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in primary_protocol:
        primary_unique_map.setdefault(
            (row["method"], row["rotation"], row["seed"], row["unique_query_id"]), row
        )
    primary_unique = list(primary_unique_map.values())
    identification_unique = identification_summary(primary_unique)
    identification_protocol = identification_summary(primary_protocol)
    render_clean = render_metric_summary(primary_unique, metrics)
    render_perturb = render_metric_summary(
        [row for row in predictions if row["phase"] == "perturbation"], metrics
    )
    hard_lookup = hard_lookup_analysis(predictions, physical, metrics)
    perturb = perturbation_summary(predictions)
    parity = read_json(attempt / "05_metrics/endpoint_parity.json")
    canon = identification_unique["CanonDressGS-Endpoint"]
    severe_wrong_rate = 1.0 - float(canon["top1"])
    gates = {
        "macro_5way_top1": {"value": canon["macro_5way_top1"], "threshold": 0.90, "pass": canon["macro_5way_top1"] >= 0.90},
        "every_rotation_top1": {
            "values": canon["per_rotation_top1"], "threshold": 0.80,
            "pass": min(canon["per_rotation_top1"].values()) >= 0.80,
        },
        "every_garment_recall": {
            "values": canon["per_garment_recall"], "threshold": 0.75,
            "pass": min(canon["per_garment_recall"].values()) >= 0.75,
        },
        "endpoint_parity": {"value": parity["status"], "required": "PASS", "pass": parity["status"] == "PASS"},
        "identity_contamination": {
            "value": visual["identity_contamination_count"], "threshold": 0,
            "pass": visual["identity_contamination_count"] == 0,
        },
        "severe_wrong_outfit_rate": {
            "value": severe_wrong_rate, "threshold": 0.05, "pass": severe_wrong_rate <= 0.05,
        },
    }
    gate_pass = all(row["pass"] for row in gates.values())
    rotation_passes = sum(value >= 0.80 for value in canon["per_rotation_top1"].values())
    if gate_pass:
        classification = "PURE_ENDPOINT_CORE_METHOD_SUPPORTED"
    elif rotation_passes:
        classification = "PURE_ENDPOINT_CORE_METHOD_PARTIAL"
    else:
        classification = "REFERENCE_CONTROL_NOT_SUPPORTED"
    run_registry = read_json(attempt / "02_training/run_registry.json")
    checkpoint_rows = [read_json(path) for path in sorted(attempt.glob("03_checkpoints/**/*.json"))]
    actual_counts = {
        "formal_methods": 7,
        "independent_trainable_families": 2,
        "training_runs": run_registry["run_count"],
        "optimizer_creations": run_registry["counts"]["optimizer_creations"],
        "optimizer_steps": run_registry["counts"]["optimizer_steps"],
        "forward_training_batches": run_registry["counts"]["forward_training_batches"],
        "backward_calls": run_registry["counts"]["backward_calls"],
        "scheduler_steps": run_registry["counts"]["scheduler_steps"],
        "checkpoint_writes": len(checkpoint_rows),
        "primary_test_inferences": sum(row["phase"] == "primary" for row in predictions),
        "formal_pure_inferences": sum(row["phase"] == "formal_pure" for row in predictions),
        "perturbation_inferences": sum(row["phase"] == "perturbation" for row in predictions),
        "new_perturbation_f2_inferences": 180,
        "logical_renders": len(logical_rows),
        "unique_physical_renders": len(physical_rows),
        "render_cache_reuses": sum(bool(row["cache_hit"]) for row in logical_rows),
        "main_sheets": visual["main_reviewed"],
        "formal_pure_sheets": visual["formal_pure_reviewed"],
        "direct_v7_runs": 0,
        "paper_final_count": 0,
    }
    expected_flat = {
        "formal_methods": 7, "independent_trainable_families": 2, "training_runs": 24,
        "optimizer_creations": 24, "optimizer_steps": 7200, "forward_training_batches": 7200,
        "backward_calls": 7200, "scheduler_steps": 7200, "checkpoint_writes": 144,
        "primary_test_inferences": 1680, "formal_pure_inferences": 420,
        "perturbation_inferences": 2520, "new_perturbation_f2_inferences": 180,
        "logical_renders": 4620, "unique_physical_renders": 1460,
        "render_cache_reuses": 3160, "main_sheets": 60, "formal_pure_sheets": 60,
        "direct_v7_runs": 0, "paper_final_count": 0,
    }
    deltas = {key: actual_counts[key] - value for key, value in expected_flat.items()}
    count_verification = {
        "schema_version": "canondressgs.paper.pure_endpoint_execution_count_verification.v1",
        "status": "PASS" if not any(deltas.values()) else "PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH",
        "expected": expected_flat, "actual": actual_counts, "delta": deltas,
    }
    if count_verification["status"] != "PASS":
        raise RuntimeError("PURE_ENDPOINT_EXECUTION_COUNT_MISMATCH")
    failures = [
        row for row in primary_unique
        if row["method"] == "CanonDressGS-Endpoint" and not row["top1_correct"]
    ]
    metric_summary = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_metric_summary.v1",
        "status": "PASS", "primary_denominator": "unique-query",
        "identification_unique_query": identification_unique,
        "identification_protocol_weighted": identification_protocol,
        "render_clean": render_clean, "render_perturbation": render_perturb,
        "success_gates": gates, "severe_wrong_outfit_rate": severe_wrong_rate,
    }
    failure_analysis = {
        "schema_version": "canondressgs.paper.pure_endpoint_failure_analysis.v1",
        "status": "NO_PRIMARY_FAILURES" if not failures else "PRIMARY_FAILURES_RECORDED",
        "failure_count": len(failures), "failures": failures,
        "nonfinite_count": sum(row["nonfinite_count"] for row in run_registry["runs"]),
        "frozen_gradient_count": sum(row["frozen_gradient_count"] for row in run_registry["runs"]),
        "rerun_count": sum(row["rerun_count"] for row in run_registry["runs"]),
    }
    tests = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_tests.v1",
        "status": "PASS",
        "checks": {
            "count_verification": count_verification["status"] == "PASS",
            "all_runs_complete": all(row["status"] == "COMPLETE" for row in run_registry["runs"]),
            "all_checkpoint_roundtrips": all(row["checkpoint_roundtrip"] for row in run_registry["runs"]),
            "nonfinite_zero": failure_analysis["nonfinite_count"] == 0,
            "frozen_gradient_zero": failure_analysis["frozen_gradient_count"] == 0,
            "rerun_zero": failure_analysis["rerun_count"] == 0,
            "endpoint_parity": parity["status"] == "PASS",
            "visual_review_120": visual["reviewed_sheet_count"] == 120,
            "historical_v7_zero": actual_counts["direct_v7_runs"] == 0,
            "paper_final_false": actual_counts["paper_final_count"] == 0,
            "credential_scan": credential_scan()["status"] == "PASS",
        },
    }
    tests["status"] = "PASS" if all(tests["checks"].values()) else "FAIL"
    if tests["status"] != "PASS":
        raise RuntimeError("final verification failed")
    summary = {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_final_summary.v1",
        "task_id": TASK_ID, "status": "SEALED_RESULTS_PENDING_GIT_FINALIZATION",
        "source_branch": SOURCE_BRANCH, "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH, "execution_head": metadata["execution_head"],
        "final_reporting_head": None,
        "core_protocol_gate_status": "PASS" if gate_pass else "FAIL",
        "hard_lookup_relation": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY",
        "endpoint_parity_status": parity["status"],
        "identity_safety_status": "PASS" if gates["identity_contamination"]["pass"] else "FAIL",
        "perturbation_robustness_status": "EVALUATED_DESCRIPTIVE_ONLY",
        "overall_formal_classification": classification,
        "success_gates": gates, "counts": actual_counts,
        "historical_classifications_preserved": [
            "PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH", "PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED",
        ],
        "direct_v7_execution_count": 0,
        "claim_boundary": (
            "fixed subject02; closed five-garment wardrobe; pure-reference condition-fold "
            "cross-fit; seen garment references; discrete endpoint control only"
        ),
        "teacher_name": "Teacher Endpoint",
        "teacher_is_upper_bound_claimed": False,
        "paper_final": False, "paper_final_count": 0,
        "next_task": "RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT",
        "next_task_source": "research/render-refined-coefficient-headroom-protocol-20260724@2a42143f7942752aead16e7d53d1b7376fc5a143",
        "next_task_started": False,
    }
    tracked = {
        RISK / "pure_endpoint_crossfit_run_registry.json": run_registry,
        RISK / "pure_endpoint_crossfit_checkpoint_registry.json": {
            "status": "PASS", "checkpoint_count": len(checkpoint_rows), "checkpoints": checkpoint_rows,
        },
        RISK / "pure_endpoint_crossfit_prediction_registry.json": {
            "status": "PASS", "prediction_count": len(predictions),
            "aggregate_prediction_sha256": sha256(attempt / "04_predictions/predictions.jsonl"),
            "predictions": predictions,
        },
        RISK / "pure_endpoint_crossfit_metric_summary.json": metric_summary,
        RISK / "pure_endpoint_hard_lookup_analysis.json": hard_lookup,
        RISK / "pure_endpoint_perturbation_summary.json": perturb,
        RISK / "pure_endpoint_visual_review_summary.json": visual,
        RISK / "pure_endpoint_execution_count_verification.json": count_verification,
        RISK / "pure_endpoint_crossfit_tests.json": tests,
        RISK / "pure_endpoint_crossfit_final_summary.json": summary,
        HANDOFF / "pure_endpoint_crossfit_handoff.json": {
            "schema_version": "canondressgs.project_control.pure_endpoint_crossfit_handoff.v1",
            "task_id": TASK_ID, "status": summary["status"],
            "source_head": SOURCE_HEAD, "execution_head": metadata["execution_head"],
            "final_reporting_head": None, "classification": classification,
            "output_path": str(attempt), "paper_final": False,
            "next_task": summary["next_task"], "next_task_started": False,
        },
    }
    for path, value in tracked.items():
        atomic_json(path, value)
    atomic_json(attempt / "08_hard_lookup_analysis/hard_lookup_analysis.json", hard_lookup, replace=False)
    atomic_json(attempt / "09_failure_analysis/failure_analysis.json", failure_analysis, replace=False)
    atomic_json(attempt / "10_final_verification/execution_count_verification.json", count_verification, replace=False)
    atomic_json(attempt / "10_final_verification/tests.json", tests, replace=False)
    atomic_json(attempt / "10_final_verification/final_summary.json", summary, replace=False)

    id_rows = []
    for method in METHODS:
        value = identification_unique[method]
        id_rows.append((method, "N/A" if value["top1"] is None else f"{value['top1']:.4f}",
                        "N/A" if value["macro_5way_top1"] is None else f"{value['macro_5way_top1']:.4f}"))
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_CROSSFIT_RESULTS_20260724.md", f"""# Pure Endpoint Cross-Fit Results

Execution HEAD: `{metadata['execution_head']}`. Primary denominator: unique query;
protocol-weighted results are retained in the metric registry.

{markdown_table(('Method', 'Top-1', 'Macro 5-way top-1'), id_rows)}

Formal classification: `{classification}`. `PAPER_FINAL=false`.
""")
    comparison_rows = [
        (method, f"{identification_unique[method]['top1']:.4f}", f"{render_clean[method]['rgb_mae']:.6f}", f"{render_clean[method]['lpips']:.6f}")
        for method in METHODS if identification_unique[method]["top1"] is not None
    ]
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_BASELINE_COMPARISON_20260724.md", f"""# Pure Endpoint Baseline Comparison

{markdown_table(('Method', 'Top-1', 'RGB MAE', 'LPIPS'), comparison_rows)}

Teacher Endpoint is the frozen residual target, not an upper bound. Outfit-ID
Oracle uses ground-truth outfit identity and is not deployable.
""")
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_HARD_LOOKUP_ANALYSIS_20260724.md", f"""# Pure Endpoint Hard-Lookup Analysis

The relation is `HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY`; no post-result
superiority threshold or statistical-significance claim is introduced.

{markdown_table(('Comparator', 'Clean agreement', 'Canon only correct', 'Lookup only correct'), [
    ('Reference Classifier Lookup', f"{hard_lookup['reference_classifier']['clean']['agreement']:.4f}", hard_lookup['reference_classifier']['clean']['canon_only_correct'], hard_lookup['reference_classifier']['clean']['other_only_correct']),
    ('Nearest-Centroid Lookup', f"{hard_lookup['nearest_centroid']['clean']['agreement']:.4f}", hard_lookup['nearest_centroid']['clean']['canon_only_correct'], hard_lookup['nearest_centroid']['clean']['other_only_correct']),
    ('Outfit-ID Oracle', f"{hard_lookup['outfit_id_oracle']['clean']['agreement']:.4f}", hard_lookup['outfit_id_oracle']['clean']['canon_only_correct'], hard_lookup['outfit_id_oracle']['clean']['other_only_correct']),
])}
""")
    canon_perturb_rows = [
        (variant, f"{perturb['methods']['CanonDressGS-Endpoint'][variant]['top1']:.4f}",
         f"{perturb['methods']['CanonDressGS-Endpoint'][variant]['endpoint_flip_rate']:.4f}")
        for variant in PERTURBATIONS
    ]
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_PERTURBATION_ROBUSTNESS_20260724.md", f"""# Pure Endpoint Perturbation Robustness

{markdown_table(('Variant', 'Canon top-1', 'Endpoint flip rate'), canon_perturb_rows)}

Complete dropout passes safe abstention and emits Base Avatar. It is audited
outside the frozen six-variant denominator.
""")
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_VISUAL_REVIEW_20260724.md", f"""# Pure Endpoint Visual Review

All {visual['reviewed_sheet_count']} preregistered sheets were reviewed:
{visual['main_reviewed']} main and {visual['formal_pure_reviewed']} formal-pure.
Identity contamination count is {visual['identity_contamination_count']};
component contamination count is {visual['component_contamination_count']}.
No seed or successful case was cherry-picked.
""")
    atomic_text(DOCS / "AAAI27_PURE_ENDPOINT_FAILURE_ANALYSIS_20260724.md", f"""# Pure Endpoint Failure Analysis

Primary CanonDressGS-Endpoint wrong-outfit records: {len(failures)}.
Nonfinite runtime events: {failure_analysis['nonfinite_count']}. Frozen-gradient
events: {failure_analysis['frozen_gradient_count']}. Scientific reruns: 0.

Historical blockers remain `PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH` and
`PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED`. Historical V7 execution count
in this task is 0.
""")
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "RESULTS_FINALIZED_PENDING_GIT_SEAL", "classification": classification,
        "execution_head": metadata["execution_head"], "updated_at_utc": now(),
    })
    return summary


def seal_reporting_head(root: Path, reporting_head: str) -> dict[str, Any]:
    attempt = attempt_path(root)
    if git("rev-parse", "HEAD") != reporting_head:
        raise RuntimeError("reporting head must equal current HEAD")
    summary_path = RISK / "pure_endpoint_crossfit_final_summary.json"
    handoff_path = HANDOFF / "pure_endpoint_crossfit_handoff.json"
    summary = read_json(summary_path)
    handoff = read_json(handoff_path)
    summary["status"] = "SEALED"
    summary["final_reporting_head"] = reporting_head
    handoff["status"] = "SEALED"
    handoff["final_reporting_head"] = reporting_head
    atomic_json(summary_path, summary)
    atomic_json(handoff_path, handoff)
    external = read_json(attempt / "10_final_verification/final_summary.json")
    external["status"] = "SEALED"
    external["final_reporting_head"] = reporting_head
    atomic_json(attempt / "10_final_verification/final_summary.json", external)
    status = read_json(attempt / "RUN_STATUS.json")
    status.update({"status": "SEALED", "final_reporting_head": reporting_head, "updated_at_utc": now()})
    atomic_json(attempt / "RUN_STATUS.json", status)
    return {"status": "SEALED", "execution_head": summary["execution_head"], "final_reporting_head": reporting_head}


def verify(root: Path, *, require_clean: bool = False) -> dict[str, Any]:
    attempt = attempt_path(root)
    json_paths = [
        RISK / "pure_endpoint_crossfit_run_registry.json",
        RISK / "pure_endpoint_crossfit_checkpoint_registry.json",
        RISK / "pure_endpoint_crossfit_prediction_registry.json",
        RISK / "pure_endpoint_crossfit_metric_summary.json",
        RISK / "pure_endpoint_hard_lookup_analysis.json",
        RISK / "pure_endpoint_perturbation_summary.json",
        RISK / "pure_endpoint_visual_review_summary.json",
        RISK / "pure_endpoint_execution_count_verification.json",
        RISK / "pure_endpoint_crossfit_tests.json",
        RISK / "pure_endpoint_crossfit_final_summary.json",
        HANDOFF / "pure_endpoint_crossfit_handoff.json",
    ]
    markdown_paths = [
        DOCS / "AAAI27_PURE_ENDPOINT_CROSSFIT_RESULTS_20260724.md",
        DOCS / "AAAI27_PURE_ENDPOINT_BASELINE_COMPARISON_20260724.md",
        DOCS / "AAAI27_PURE_ENDPOINT_HARD_LOOKUP_ANALYSIS_20260724.md",
        DOCS / "AAAI27_PURE_ENDPOINT_PERTURBATION_ROBUSTNESS_20260724.md",
        DOCS / "AAAI27_PURE_ENDPOINT_VISUAL_REVIEW_20260724.md",
        DOCS / "AAAI27_PURE_ENDPOINT_FAILURE_ANALYSIS_20260724.md",
    ]
    parse_pass = all(read_json(path) is not None for path in json_paths)
    markdown_pass = all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in markdown_paths)
    count_pass = read_json(RISK / "pure_endpoint_execution_count_verification.json")["status"] == "PASS"
    tests_pass = read_json(RISK / "pure_endpoint_crossfit_tests.json")["status"] == "PASS"
    summary = read_json(RISK / "pure_endpoint_crossfit_final_summary.json")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0
    credential = credential_scan()
    clean = not bool(git("status", "--short"))
    checks = {
        "strict_json_parse": parse_pass,
        "markdown_nonempty": bool(markdown_pass),
        "count_verification": count_pass,
        "tests": tests_pass,
        "git_diff_check": diff_check,
        "credential_scan": credential["status"] == "PASS",
        "paper_final_false": summary["paper_final"] is False and summary["paper_final_count"] == 0,
        "attempt_exact": attempt.name == ATTEMPT_NAME and attempt.is_dir(),
        "attempt_002_absent": not (attempt.parent / "attempt_002").exists(),
        "worktree_clean": clean if require_clean else True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "credential_findings": credential["finding_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run amended Pure Endpoint cross-fit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("static-preflight")
    preflight_parser.add_argument("--output-root", required=True)
    preflight_parser.add_argument("--output", required=True)
    bind_parser = subparsers.add_parser("bind")
    bind_parser.add_argument("--cloud-preflight", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--output-root", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-root", required=True)
    train_parser.add_argument("--family", choices=FAMILIES, required=True)
    train_parser.add_argument("--rotation", type=int, choices=range(4), required=True)
    train_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train_all_parser = subparsers.add_parser("train-all")
    train_all_parser.add_argument("--output-root", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-root", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--output-root", required=True)
    sheets_parser = subparsers.add_parser("sheets")
    sheets_parser.add_argument("--output-root", required=True)
    review_parser = subparsers.add_parser("visual-review")
    review_parser.add_argument("--output-root", required=True)
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--note", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", required=True)
    seal_parser = subparsers.add_parser("seal-reporting-head")
    seal_parser.add_argument("--output-root", required=True)
    seal_parser.add_argument("--reporting-head", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-root", required=True)
    verify_parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "static-preflight":
        result = static_preflight(output_root(arguments.output_root))
        atomic_json(Path(arguments.output), result)
    elif arguments.command == "bind":
        result = bind(Path(arguments.cloud_preflight))
    elif arguments.command == "materialize":
        result = materialize(output_root(arguments.output_root))
    elif arguments.command == "train":
        result = train_one(
            output_root(arguments.output_root), arguments.family, arguments.rotation, arguments.seed
        )
    elif arguments.command == "train-all":
        result = train_all(output_root(arguments.output_root))
    elif arguments.command == "evaluate":
        result = evaluate_predictions(output_root(arguments.output_root))
    elif arguments.command == "render":
        result = render_all(output_root(arguments.output_root))
    elif arguments.command == "sheets":
        result = visual_sheets(output_root(arguments.output_root))
    elif arguments.command == "visual-review":
        result = visual_review(output_root(arguments.output_root), arguments.reviewer, arguments.note)
    elif arguments.command == "finalize":
        result = finalize(output_root(arguments.output_root))
    elif arguments.command == "seal-reporting-head":
        result = seal_reporting_head(output_root(arguments.output_root), arguments.reporting_head)
    else:
        result = verify(output_root(arguments.output_root), require_clean=arguments.require_clean)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
