#!/usr/bin/env python3
"""Append sealed Pure Endpoint evidence to the paper figure bank.

The tool reads only explicitly supplied sealed inputs. It never discovers or
opens Headroom outputs, training directories, or checkpoint files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


FIGURE_BANK_HEAD = "06261aea35d26006db250df65707d1a09a0bf098"
RESULT_HEAD = "ce110887a942cf8db082ba688c8d36d2433bfdbe"
EXECUTION_HEAD = "195fb887f2cac8a72920b499c44bb66700a97e25"
RESULT_BRANCH = "research/pure-endpoint-core-method-crossfit-amended-20260724"
TASK_ID = "AAAI27-PAPER-FIGURE-BANK-PURE-ENDPOINT-REFRESH-001"
SOURCE_TASK_ID = "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"
GARMENTS = ["O01", "O02", "O03", "O04", "O08"]
METHODS = [
    "Base Avatar",
    "Teacher Endpoint",
    "Outfit-ID Oracle",
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Linear Coefficient Predictor",
    "CanonDressGS-Endpoint",
]
SOURCE_REPO_FILES = [
    "docs/PAPER/AAAI27_PURE_ENDPOINT_CROSSFIT_RESULTS_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_BASELINE_COMPARISON_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_HARD_LOOKUP_ANALYSIS_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_PERTURBATION_ROBUSTNESS_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_VISUAL_REVIEW_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_FAILURE_ANALYSIS_20260724.md",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_run_registry.json",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_checkpoint_registry.json",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_prediction_registry.json",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_metric_summary.json",
    "paper_protocol/reviewer_risk/pure_endpoint_hard_lookup_analysis.json",
    "paper_protocol/reviewer_risk/pure_endpoint_perturbation_summary.json",
    "paper_protocol/reviewer_risk/pure_endpoint_visual_review_summary.json",
    "paper_protocol/reviewer_risk/pure_endpoint_execution_count_verification.json",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_tests.json",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_final_summary.json",
    "project_control_handoff/pure_endpoint_crossfit_handoff.json",
]
SOURCE_OUTPUT_FILES = [
    "04_predictions/complete_dropout_safety_audit.json",
    "04_predictions/perturbed_reference_feature_rows.json",
    "04_predictions/prediction_manifest.json",
    "05_metrics/endpoint_parity.json",
    "06_renders/logical_render_registry.jsonl",
    "06_renders/physical_render_registry.jsonl",
    "06_renders/render_manifest.json",
    "07_visual_sheets/selection_manifest.json",
    "07_visual_sheets/visual_review.json",
    "08_hard_lookup_analysis/hard_lookup_analysis.json",
    "09_failure_analysis/failure_analysis.json",
    "10_final_verification/execution_count_verification.json",
    "10_final_verification/final_summary.json",
    "10_final_verification/tests.json",
]
OUTPUT_DIRS = [
    "00_preflight",
    "01_source_snapshot",
    "02_pure_endpoint_inventory",
    "03_contact_sheets",
    "04_metric_plots",
    "05_candidate_panels",
    "06_provenance",
    "07_claim_audit",
    "08_final_verification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "artifacts", "verify", "all"), default="all")
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-tool", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r} in {path}")
            value[key] = item
        return value

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"idempotence conflict: {path}")
        return
    path.write_bytes(content)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json_bytes(value))


def copy_file(source: Path, target: Path) -> None:
    content = source.read_bytes()
    write_bytes(target, content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_asset_id(kind: str, path: str) -> str:
    token = hashlib.sha256(f"{kind}|{path}".encode("utf-8")).hexdigest()[:16].upper()
    return f"PE-{kind}-{token}"


def check_source(source_repo: Path, source_attempt: Path) -> Dict[str, Any]:
    summary = load_json(source_repo / "paper_protocol/reviewer_risk/pure_endpoint_crossfit_final_summary.json")
    counts = load_json(
        source_repo / "paper_protocol/reviewer_risk/pure_endpoint_execution_count_verification.json"
    )
    tests = load_json(source_repo / "paper_protocol/reviewer_risk/pure_endpoint_crossfit_tests.json")
    required_counts = {
        "formal_methods": 7,
        "independent_trainable_families": 2,
        "training_runs": 24,
        "optimizer_steps": 7200,
        "checkpoint_writes": 144,
        "logical_renders": 4620,
        "unique_physical_renders": 1460,
        "render_cache_reuses": 3160,
        "main_sheets": 60,
        "formal_pure_sheets": 60,
    }
    failures: List[str] = []
    if summary.get("status") != "SEALED":
        failures.append("FINAL_SUMMARY_NOT_SEALED")
    if summary.get("overall_formal_classification") != "PURE_ENDPOINT_CORE_METHOD_SUPPORTED":
        failures.append("FINAL_CLASSIFICATION_MISMATCH")
    if summary.get("execution_head") != EXECUTION_HEAD:
        failures.append("EXECUTION_HEAD_MISMATCH")
    if counts.get("status") != "PASS" or any(value != 0 for value in counts.get("delta", {}).values()):
        failures.append("EXECUTION_COUNT_VERIFICATION_FAILED")
    if tests.get("status") != "PASS":
        failures.append("SOURCE_TESTS_FAILED")
    for key, expected in required_counts.items():
        if counts.get("actual", {}).get(key) != expected:
            failures.append(f"COUNT_MISMATCH:{key}")
    if not source_attempt.is_dir():
        failures.append("ATTEMPT_001_MISSING")
    if source_attempt.with_name("attempt_002").exists():
        failures.append("ATTEMPT_002_PRESENT")
    missing = [item for item in SOURCE_REPO_FILES if not (source_repo / item).is_file()]
    missing += [item for item in SOURCE_OUTPUT_FILES if not (source_attempt / item).is_file()]
    failures += [f"MISSING_SOURCE:{item}" for item in missing]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "required_counts": required_counts,
        "source_summary": {
            "classification": summary.get("overall_formal_classification"),
            "execution_head": summary.get("execution_head"),
            "hard_lookup_relation": summary.get("hard_lookup_relation"),
            "endpoint_parity_status": summary.get("endpoint_parity_status"),
            "identity_safety_status": summary.get("identity_safety_status"),
            "perturbation_robustness_status": summary.get("perturbation_robustness_status"),
        },
    }


def source_data(source_repo: Path, source_attempt: Path) -> Dict[str, Any]:
    predictions = load_json(
        source_repo / "paper_protocol/reviewer_risk/pure_endpoint_crossfit_prediction_registry.json"
    )["predictions"]
    physical = load_jsonl(source_attempt / "06_renders/physical_render_registry.jsonl")
    logical = load_jsonl(source_attempt / "06_renders/logical_render_registry.jsonl")
    if len(predictions) != 4620 or len(logical) != 4620 or len(physical) != 1460:
        raise RuntimeError("sealed prediction/render counts do not match")
    for index, (prediction, render) in enumerate(zip(predictions, logical)):
        keys = ("method", "phase", "rotation", "seed", "variant", "render_signature")
        if any(prediction.get(key) != render.get(key) for key in keys):
            raise RuntimeError(f"prediction/logical render mismatch at {index}")
    return {
        "predictions": predictions,
        "physical": physical,
        "logical": logical,
        "selection": load_json(source_attempt / "07_visual_sheets/selection_manifest.json"),
        "dropout": load_json(source_attempt / "04_predictions/complete_dropout_safety_audit.json"),
        "parity": load_json(source_attempt / "05_metrics/endpoint_parity.json"),
        "metrics": load_json(
            source_repo / "paper_protocol/reviewer_risk/pure_endpoint_crossfit_metric_summary.json"
        ),
        "hard_lookup": load_json(
            source_repo / "paper_protocol/reviewer_risk/pure_endpoint_hard_lookup_analysis.json"
        ),
        "perturbation": load_json(
            source_repo / "paper_protocol/reviewer_risk/pure_endpoint_perturbation_summary.json"
        ),
        "visual_review": load_json(
            source_repo / "paper_protocol/reviewer_risk/pure_endpoint_visual_review_summary.json"
        ),
        "final_summary": load_json(
            source_repo / "paper_protocol/reviewer_risk/pure_endpoint_crossfit_final_summary.json"
        ),
    }


def inventory_assets(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logical_by_physical: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for logical, prediction in zip(data["logical"], data["predictions"]):
        logical_by_physical[int(logical["physical_index"])].append((logical, prediction))
    assets: List[Dict[str, Any]] = []
    size_by_sha: Dict[str, int] = {}
    for physical in data["physical"]:
        links = logical_by_physical[int(physical["physical_index"])]
        query_ids = sorted({item[0]["query_id"] for item in links})
        unique_query_ids = sorted({item[0]["unique_query_id"] for item in links})
        checkpoints = sorted({item[1]["checkpoint_sha256"] for item in links})
        endpoints = sorted({str(item[1].get("realized_endpoint")) for item in links})
        for channel in ("rgb", "alpha"):
            path = str(physical[f"{channel}_path"])
            source_sha = str(physical[f"{channel}_sha256"])
            size = Path(path).stat().st_size
            size_by_sha.setdefault(source_sha, size)
            assets.append({
                "asset_id": make_asset_id(f"RENDER-{channel.upper()}", path),
                "asset_type": "SEALED_PURE_ENDPOINT_SCIENTIFIC_RENDER",
                "channel": channel.upper(),
                "source_category": "SEALED_CURRENT_ENDPOINT_EVIDENCE",
                "result_branch": RESULT_BRANCH,
                "source_head": RESULT_HEAD,
                "execution_head": EXECUTION_HEAD,
                "attempt_id": "attempt_001",
                "experiment_id": SOURCE_TASK_ID,
                "method": physical["method"],
                "rotation": physical["rotation"],
                "seed": physical["seed"],
                "garment": physical["gt_garment"],
                "query_id": query_ids[0] if len(query_ids) == 1 else "MULTIPLE_LOGICAL_QUERIES",
                "logical_query_ids": query_ids,
                "unique_query_ids": unique_query_ids,
                "perturbation": physical["variant"],
                "endpoint": endpoints[0] if len(endpoints) == 1 else "MULTIPLE_OR_NOT_APPLICABLE",
                "checkpoint_sha256": checkpoints[0] if len(checkpoints) == 1 else checkpoints,
                "render_sha256": source_sha,
                "renderer_sha256": physical["renderer_source_sha256_lf"],
                "original_path": path,
                "original_sha256": source_sha,
                "original_bytes": size,
                "transform": None,
                "figure_eligibility": "SEALED_CURRENT_ENDPOINT_EVIDENCE",
                "claim_boundary": "CLOSED_WARDROBE_DISCRETE_ENDPOINT_ONLY",
            })
    for item in data["selection"]["items"]:
        path = str(item["path"])
        source_sha = sha256_file(Path(path))
        size = Path(path).stat().st_size
        size_by_sha.setdefault(source_sha, size)
        assets.append({
            "asset_id": make_asset_id("VISUAL-SHEET", path),
            "asset_type": "SEALED_PURE_ENDPOINT_VISUAL_SHEET",
            "source_category": "SEALED_CURRENT_ENDPOINT_EVIDENCE",
            "result_branch": RESULT_BRANCH,
            "source_head": RESULT_HEAD,
            "execution_head": EXECUTION_HEAD,
            "attempt_id": "attempt_001",
            "experiment_id": SOURCE_TASK_ID,
            "method": "SEVEN_METHOD_COMPARISON_SHEET",
            "rotation": item["rotation"],
            "seed": item["seed"],
            "garment": item["garment"],
            "query_id": item["condition"],
            "unique_query_ids": [item["unique_query_id"]],
            "perturbation": "clean",
            "endpoint": item["garment"],
            "checkpoint_sha256": "SEE_RENDER_SIGNATURES_AND_SOURCE_REGISTRY",
            "render_sha256": item["artifact_hashes"],
            "original_path": path,
            "original_sha256": source_sha,
            "original_bytes": size,
            "transform": None,
            "figure_eligibility": "REQUIRES_MANUAL_ADJUDICATION",
            "claim_boundary": "REVIEW_SHEET_NOT_FINAL_PAPER_FIGURE",
            "sheet_category": item["category"],
        })
    for path_value in data["selection"]["overview_paths"]:
        path = str(path_value)
        source_sha = sha256_file(Path(path))
        size = Path(path).stat().st_size
        size_by_sha.setdefault(source_sha, size)
        assets.append({
            "asset_id": make_asset_id("OVERVIEW", path),
            "asset_type": "SEALED_PURE_ENDPOINT_VISUAL_SHEET",
            "source_category": "SEALED_CURRENT_ENDPOINT_EVIDENCE",
            "result_branch": RESULT_BRANCH,
            "source_head": RESULT_HEAD,
            "execution_head": EXECUTION_HEAD,
            "attempt_id": "attempt_001",
            "experiment_id": SOURCE_TASK_ID,
            "method": "OVERVIEW",
            "rotation": "MULTIPLE",
            "seed": "MULTIPLE",
            "garment": "MULTIPLE",
            "query_id": "MULTIPLE",
            "unique_query_ids": [],
            "perturbation": "clean",
            "endpoint": "MULTIPLE",
            "checkpoint_sha256": "SEE_SOURCE_REGISTRY",
            "render_sha256": "SEE_SELECTION_MANIFEST",
            "original_path": path,
            "original_sha256": source_sha,
            "original_bytes": size,
            "transform": None,
            "figure_eligibility": "SUPPLEMENTARY_PARITY_SAFETY_CANDIDATE",
            "claim_boundary": "OVERVIEW_NOT_FINAL_PAPER_FIGURE",
        })
    sha_counts = Counter(item["original_sha256"] for item in assets)
    total_bytes = sum(int(item["original_bytes"]) for item in assets)
    unique_bytes = sum(size_by_sha.values())
    statistics = {
        "source_visual_count": len(assets),
        "source_unique_sha": len(sha_counts),
        "source_duplicate_records": len(assets) - len(sha_counts),
        "source_visual_bytes": total_bytes,
        "unique_source_bytes": unique_bytes,
        "duplicate_avoided_bytes": total_bytes - unique_bytes,
        "asset_type_counts": dict(sorted(Counter(item["asset_type"] for item in assets).items())),
    }
    return sorted(assets, key=lambda item: item["asset_id"]), statistics


def snapshot_sources(source_repo: Path, source_attempt: Path, output_root: Path) -> None:
    repo_root = output_root / "01_source_snapshot/repo"
    attempt_root = output_root / "01_source_snapshot/output"
    for relative in SOURCE_REPO_FILES:
        copy_file(source_repo / relative, repo_root / relative)
    for relative in SOURCE_OUTPUT_FILES:
        copy_file(source_attempt / relative, attempt_root / relative)


def run_inventory(source_repo: Path, source_attempt: Path, output_root: Path) -> Dict[str, Any]:
    gate = check_source(source_repo, source_attempt)
    if gate["status"] != "PASS":
        raise RuntimeError(json.dumps(gate, sort_keys=True))
    for dirname in OUTPUT_DIRS:
        (output_root / dirname).mkdir(parents=True, exist_ok=True)
    snapshot_sources(source_repo, source_attempt, output_root)
    data = source_data(source_repo, source_attempt)
    assets, statistics = inventory_assets(data)
    registry = {
        "schema_version": "paper_figure_asset_registry_pure_endpoint_refresh.v1",
        "task_id": TASK_ID,
        "source_branch": RESULT_BRANCH,
        "source_head": RESULT_HEAD,
        "execution_head": EXECUTION_HEAD,
        "attempt_id": "attempt_001",
        "classification": "PURE_ENDPOINT_CORE_METHOD_SUPPORTED",
        "assets": assets,
        "statistics": statistics,
        "headroom_active_output_files_read": 0,
        "avatarrex_media_exports": 0,
    }
    write_json(output_root / "00_preflight/source_integrity_gate.json", gate)
    write_json(output_root / "02_pure_endpoint_inventory/pure_endpoint_source_visual_registry.json", registry)
    write_json(output_root / "02_pure_endpoint_inventory/source_statistics.json", statistics)
    write_json(output_root / "02_pure_endpoint_inventory/selection_manifest_snapshot.json", data["selection"])
    return {"gate": gate, "statistics": statistics}


def is_safe_dropout_abstention(row: Dict[str, Any]) -> bool:
    return (
        row.get("status") == "ABSTAIN_EMPTY_REFERENCE"
        and row.get("realization") == "Base Avatar"
        and row.get("garment_endpoint_selected") is False
        and row.get("valid_reference_count") == 0
    )


def build_plot_source(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = data["metrics"]["identification_unique_query"]
    hard = data["hard_lookup"]
    perturb = data["perturbation"]
    source_files = {
        "metrics": "paper_protocol/reviewer_risk/pure_endpoint_crossfit_metric_summary.json",
        "hard_lookup": "paper_protocol/reviewer_risk/pure_endpoint_hard_lookup_analysis.json",
        "perturbation": "paper_protocol/reviewer_risk/pure_endpoint_perturbation_summary.json",
        "parity": "05_metrics/endpoint_parity.json",
        "dropout": "04_predictions/complete_dropout_safety_audit.json",
        "render_manifest": "06_renders/render_manifest.json",
        "visual_review": "paper_protocol/reviewer_risk/pure_endpoint_visual_review_summary.json",
    }
    method_order = METHODS
    plots: List[Dict[str, Any]] = []
    top1_values = [
        {"x": index, "y": metrics[method]["top1"]}
        for index, method in enumerate(method_order)
        if metrics[method]["top1"] is not None
    ]
    plots.append({
        "plot_id": "clean_method_top1",
        "kind": "bar",
        "title": "Clean endpoint top-1 (n=60 per evaluated method)",
        "x_label": "Method",
        "y_label": "Top-1",
        "y_limits": [0.0, 1.05],
        "x_ticks": [{"value": i, "label": method} for i, method in enumerate(method_order)],
        "series": [{"label": "Top-1", "values": top1_values}],
        "not_applicable": {"Base Avatar": "NO_ENDPOINT_PREDICTION"},
        "denominator": 60,
        "source_files": [source_files["metrics"]],
        "source_json_pointers": ["/identification_unique_query/*/top1"],
        "claim_status": "CLOSED_WARDROBE_DESCRIPTIVE_ONLY",
    })
    endpoint_methods = method_order
    plots.append({
        "plot_id": "endpoint_exact_match",
        "kind": "bar",
        "title": "Endpoint exact match (n=60)",
        "x_label": "Method",
        "y_label": "Exact match",
        "y_limits": [0.0, 1.05],
        "x_ticks": [{"value": i, "label": method} for i, method in enumerate(endpoint_methods)],
        "series": [{"label": "Exact match", "values": [
            {"x": i, "y": metrics[method].get("endpoint_exact_match")}
            for i, method in enumerate(endpoint_methods)
            if metrics[method].get("endpoint_exact_match") is not None
        ]}],
        "not_applicable": {
            "Base Avatar": "NO_ENDPOINT_PREDICTION",
            "Teacher Endpoint": "TARGET_ENDPOINT_NOT_A_PREDICTION_FIELD",
        },
        "denominator": 60,
        "source_files": [source_files["metrics"]],
        "source_json_pointers": ["/identification_unique_query/*/endpoint_exact_match"],
        "claim_status": "RAW_AND_REALIZED_SEMANTICS_SEPARATED",
    })
    comparisons = ["reference_classifier", "nearest_centroid", "outfit_id_oracle"]
    labels = ["Reference classifier", "Nearest centroid", "Outfit-ID oracle"]
    parity_rows = data["parity"]["rows"]
    renderer_input_exact = all(row["renderer_input_exact"] is True for row in parity_rows)
    physical_render_exact = all(
        row["physical_render_parity"] == "SAME_INPUT_SAME_CACHE_SIGNATURE"
        for row in parity_rows
    )
    equivalence_labels = labels + ["Renderer input exact", "Physical render parity"]
    plots.append({
        "plot_id": "hard_lookup_agreement_clean",
        "kind": "bar",
        "title": "Clean functional equivalence",
        "x_label": "Registered equivalence check",
        "y_label": "Exact fraction",
        "y_limits": [0.0, 1.05],
        "x_ticks": [{"value": i, "label": label} for i, label in enumerate(equivalence_labels)],
        "series": [{"label": "Agreement", "values": [
            {"x": i, "y": hard[key]["clean"]["agreement"]}
            for i, key in enumerate(comparisons)
        ] + [
            {"x": 3, "y": 1.0 if renderer_input_exact else 0.0},
            {"x": 4, "y": 1.0 if physical_render_exact else 0.0},
        ]}],
        "denominators": {"endpoint_agreement_per_comparison": 60, "parity_rows": len(parity_rows)},
        "source_files": [source_files["hard_lookup"], source_files["parity"]],
        "source_json_pointers": [f"/{key}/clean" for key in comparisons] + ["/rows"],
        "claim_status": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY",
    })
    overlap_fields = ["both_correct", "canon_only_correct", "other_only_correct", "both_wrong"]
    plots.append({
        "plot_id": "hard_lookup_error_overlap_perturbation",
        "kind": "bar",
        "title": "Perturbation outcome overlap (n=360 per comparison)",
        "x_label": "Comparison",
        "y_label": "Count",
        "x_ticks": [{"value": i, "label": label} for i, label in enumerate(labels)],
        "series": [{
            "label": field.replace("_", " "),
            "values": [{"x": i, "y": hard[key]["perturbation"][field]} for i, key in enumerate(comparisons)],
        } for field in overlap_fields],
        "denominator": 360,
        "source_files": [source_files["hard_lookup"]],
        "source_json_pointers": [f"/{key}/perturbation" for key in comparisons],
        "claim_status": "DESCRIPTIVE_COUNTS_NO_SUPERIORITY_CLAIM",
    })
    variants = sorted(perturb["methods"]["CanonDressGS-Endpoint"])
    flip_variants = ["clean", *variants, "complete_dropout"]
    flip_methods = ["CanonDressGS-Endpoint", "Reference Classifier Lookup", "Nearest-Centroid Lookup"]
    plots.append({
        "plot_id": "endpoint_flip_by_perturbation",
        "kind": "bar",
        "title": "Endpoint flip by registered input condition",
        "x_label": "Perturbation",
        "y_label": "Flip rate",
        "y_limits": [0.0, 1.0],
        "x_ticks": [{"value": i, "label": variant} for i, variant in enumerate(flip_variants)],
        "series": [{
            "label": method,
            "values": [
                {"x": 0, "y": 1.0 - metrics[method]["endpoint_exact_match"]},
                *[
                    {"x": i + 1, "y": perturb["methods"][method][variant]["endpoint_flip_rate"]}
                    for i, variant in enumerate(variants)
                ],
            ],
        } for method in flip_methods],
        "not_applicable": {"complete_dropout": "SAFE_ABSTENTION_NO_ENDPOINT_PREDICTION"},
        "denominators": {
            "clean_per_method": 60,
            "registered_perturbation_per_cell": 60,
            "complete_dropout_per_method": 20,
        },
        "source_files": [source_files["metrics"], source_files["perturbation"], source_files["dropout"]],
        "source_json_pointers": [
            "/identification_unique_query/*/endpoint_exact_match",
            "/methods/*/*/endpoint_flip_rate",
            "/records",
        ],
        "claim_status": "PERTURBATION_ROBUSTNESS_EVALUATED_DESCRIPTIVE_ONLY",
    })
    shared_raw_mae = metrics["Linear Coefficient Predictor"]["coefficient_mae"]
    plots.append({
        "plot_id": "raw_vs_realized_coefficient_error",
        "kind": "bar",
        "title": "Shared raw predictor vs realized coefficient error (n=60)",
        "x_label": "Realization",
        "y_label": "Coefficient MAE",
        "x_ticks": [
            {"value": 0, "label": "Linear continuous\nendpoint exact=0"},
            {"value": 1, "label": "Canon endpoint snap\nendpoint exact=1"},
        ],
        "series": [
            {"label": "RAW_COEFFICIENT_ERROR", "values": [
                {"x": 0, "y": shared_raw_mae}, {"x": 1, "y": shared_raw_mae},
            ]},
            {"label": "REALIZED_ENDPOINT_ERROR", "values": [
                {"x": 0, "y": shared_raw_mae},
                {"x": 1, "y": metrics["CanonDressGS-Endpoint"]["coefficient_mae"]},
            ]},
        ],
        "endpoint_exact_match": {"Linear continuous": 0.0, "Canon endpoint snap": 1.0},
        "shared_predictor_family_source": "Linear Coefficient Predictor sealed raw prediction",
        "denominator": 60,
        "source_files": [source_files["metrics"]],
        "source_json_pointers": [
            "/identification_unique_query/Linear Coefficient Predictor/coefficient_mae",
            "/identification_unique_query/CanonDressGS-Endpoint/coefficient_mae",
        ],
        "claim_status": "RAW_PREDICTION_IS_NOT_EXACT_ENDPOINT_SUCCESS_USES_SNAPPING",
    })
    dropout_count = sum(is_safe_dropout_abstention(row) for row in data["dropout"]["records"])
    reviewed = data["visual_review"]
    safety_labels = ["Identity contamination", "Component contamination", "Severe wrong outfit", "Dropout abstention"]
    safety_values = [
        reviewed["identity_contamination_count"],
        reviewed["component_contamination_count"],
        reviewed["severe_sheet_count"],
        dropout_count / data["dropout"]["record_count"],
    ]
    plots.append({
        "plot_id": "identity_and_safety_summary",
        "kind": "bar",
        "title": "Identity and registered safety checks",
        "x_label": "Check",
        "y_label": "Count or registered rate",
        "x_ticks": [{"value": i, "label": label} for i, label in enumerate(safety_labels)],
        "series": [{"label": "Observed", "values": [{"x": i, "y": value} for i, value in enumerate(safety_values)]}],
        "denominators": {"visual_sheets": 120, "dropout": data["dropout"]["record_count"]},
        "source_files": [source_files["visual_review"], source_files["dropout"]],
        "source_json_pointers": ["/identity_contamination_count", "/component_contamination_count", "/records"],
        "claim_status": "ENGINEERING_SAFETY_EVIDENCE_NOT_GENERALIZATION",
    })
    final_counts = data["final_summary"]["counts"]
    parity_pass = renderer_input_exact and physical_render_exact
    plots.append({
        "plot_id": "render_equivalence_and_cache_summary",
        "kind": "bar",
        "title": "Renderer equivalence and cache audit",
        "x_label": "Audit item",
        "y_label": "Fraction",
        "y_limits": [0.0, 1.05],
        "x_ticks": [
            {"value": 0, "label": "Renderer input exact"},
            {"value": 1, "label": "Physical render parity"},
            {"value": 2, "label": "Logical cache reuse"},
        ],
        "series": [{"label": "Fraction", "values": [
            {"x": 0, "y": 1.0 if parity_pass else 0.0},
            {"x": 1, "y": 1.0 if parity_pass else 0.0},
            {"x": 2, "y": final_counts["render_cache_reuses"] / final_counts["logical_renders"]},
        ]}],
        "counts": {
            "logical_renders": final_counts["logical_renders"],
            "unique_physical_renders": final_counts["unique_physical_renders"],
            "render_cache_reuses": final_counts["render_cache_reuses"],
        },
        "source_files": [source_files["parity"], source_files["render_manifest"]],
        "source_json_pointers": ["/rows", "/logical_renders", "/render_cache_reuses"],
        "claim_status": "SAME_ENDPOINT_RENDER_PARITY_NOT_IMAGE_QUALITY_SUPERIORITY",
    })
    return {
        "schema_version": "paper_figure_pure_endpoint_plot_source.v1",
        "task_id": TASK_ID,
        "source_head": RESULT_HEAD,
        "execution_head": EXECUTION_HEAD,
        "plots": plots,
    }


def logical_entries(data: Dict[str, Any], asset_by_path: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    physical = {int(row["physical_index"]): row for row in data["physical"]}
    entries: List[Dict[str, Any]] = []
    for logical, prediction in zip(data["logical"], data["predictions"]):
        source = physical[int(logical["physical_index"])]
        asset = asset_by_path[str(source["rgb_path"])]
        entries.append({
            "asset_id": asset["asset_id"],
            "source_path": str(source["rgb_path"]),
            "source_sha256": source["rgb_sha256"],
            "method": logical["method"],
            "phase": logical["phase"],
            "rotation": logical["rotation"],
            "seed": logical["seed"],
            "garment": prediction["gt_garment"],
            "query_id": logical["query_id"],
            "unique_query_id": logical["unique_query_id"],
            "perturbation": logical["variant"],
            "predicted_endpoint": prediction.get("predicted_garment"),
            "realized_endpoint": prediction.get("realized_endpoint"),
            "correct_endpoint": prediction.get("target_endpoint"),
            "endpoint_exact_match": prediction.get("exact_endpoint_match"),
            "checkpoint_sha256": prediction.get("checkpoint_sha256"),
            "figure_eligibility": "SEALED_CURRENT_ENDPOINT_EVIDENCE",
        })
    return entries


def image_modules():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


def save_image(path: Path, image: Any, image_format: str = "PNG") -> None:
    import io
    buffer = io.BytesIO()
    save_args: Dict[str, Any] = {"format": image_format}
    if image_format == "PNG":
        save_args.update({"compress_level": 9, "optimize": False})
    image.save(buffer, **save_args)
    write_bytes(path, buffer.getvalue())


def build_sheet(
    name: str,
    entries: Sequence[Dict[str, Any]],
    output: Path,
    columns: int = 5,
    image_box: Tuple[int, int] = (180, 180),
    selection_rule: str = "FROZEN_REGISTERED_ORDER",
    denominator: Any = None,
) -> Dict[str, Any]:
    Image, ImageDraw, ImageFont = image_modules()
    font = ImageFont.load_default()
    title_height = 52
    label_height = 104
    cell_width = image_box[0] + 20
    cell_height = image_box[1] + label_height + 20
    rows = max(1, math.ceil(max(1, len(entries)) / columns))
    canvas = Image.new("RGB", (columns * cell_width + 16, rows * cell_height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{name} | REVIEW ONLY | NOT PAPER_FINAL", fill="black", font=font)
    draw.text((10, 25), f"selection={selection_rule} denominator={denominator}", fill=(100, 0, 0), font=font)
    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        x = 8 + column * cell_width
        y = title_height + row * cell_height
        if entry.get("source_path"):
            source_path = Path(entry["source_path"])
            actual_source_sha = sha256_file(source_path)
            if actual_source_sha != entry["source_sha256"]:
                raise RuntimeError(
                    f"selected source SHA mismatch: {source_path}: "
                    f"{actual_source_sha} != {entry['source_sha256']}"
                )
            with Image.open(source_path) as source:
                source.load()
                tile = source.convert("RGB")
                tile.thumbnail(image_box, Image.Resampling.LANCZOS)
            px = x + 10 + (image_box[0] - tile.width) // 2
            py = y + 8 + (image_box[1] - tile.height) // 2
            canvas.paste(tile, (px, py))
        else:
            message = entry.get("placeholder_text", "SOURCE_VISUAL_NOT_AVAILABLE")
            draw.multiline_text(
                (x + 22, y + image_box[1] // 2), textwrap.fill(message, width=22),
                fill=(110, 0, 0), font=font, spacing=3,
            )
        draw.rectangle((x + 9, y + 7, x + 10 + image_box[0], y + 8 + image_box[1]), outline=(90, 90, 90))
        lines = [
            entry.get("asset_id", "NO_ASSET_ID"),
            f"g={entry.get('garment')} r={entry.get('rotation')} s={entry.get('seed')}",
            f"m={entry.get('method')} p={entry.get('perturbation')}",
            f"q={entry.get('query_id')}",
            f"pred={entry.get('realized_endpoint') or entry.get('predicted_endpoint')} gt={entry.get('correct_endpoint')}",
            f"sha={entry.get('source_sha256', '')[:12]} elig={entry.get('figure_eligibility')}",
        ]
        label = "\n".join(
            wrapped
            for line in lines
            for wrapped in textwrap.wrap(str(line), width=34, break_long_words=True) or [""]
        )
        draw.multiline_text((x + 10, y + image_box[1] + 12), label, fill="black", font=font, spacing=1)
    save_image(output, canvas)
    return {
        "transform_id": f"PE-SHEET-{name.upper().replace('_', '-')}",
        "operation": "DETERMINISTIC_LANCZOS_THUMBNAIL_COMPOSITION",
        "parameters": {"columns": columns, "image_box": list(image_box), "selection_rule": selection_rule},
        "input_asset_ids": [entry.get("asset_id") for entry in entries if entry.get("source_path")],
        "input_sha256": [entry.get("source_sha256") for entry in entries if entry.get("source_path")],
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "crop": None,
        "asymmetric_crop": False,
        "method_specific_enhancement": False,
        "ai_generated": False,
    }


def compose_plots(name: str, plot_paths: Sequence[Path], output: Path, columns: int = 2) -> Dict[str, Any]:
    Image, ImageDraw, ImageFont = image_modules()
    font = ImageFont.load_default()
    box = (960, 570)
    rows = math.ceil(len(plot_paths) / columns)
    canvas = Image.new("RGB", (columns * box[0], rows * box[1] + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{name} | CANDIDATE | REQUIRES MANUAL ADJUDICATION", fill=(100, 0, 0), font=font)
    for index, path in enumerate(plot_paths):
        with Image.open(path) as source:
            source.load()
            tile = source.convert("RGB")
            tile.thumbnail((box[0] - 20, box[1] - 20), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * box[0] + (box[0] - tile.width) // 2
        y = 42 + row * box[1] + (box[1] - tile.height) // 2
        canvas.paste(tile, (x, y))
    save_image(output, canvas)
    return {
        "transform_id": f"PE-CANDIDATE-{name.upper().replace('_', '-')}",
        "operation": "DETERMINISTIC_PLOT_PANEL_COMPOSITION",
        "parameters": {"columns": columns, "box": list(box)},
        "input_paths": [str(path) for path in plot_paths],
        "input_sha256": [sha256_file(path) for path in plot_paths],
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "crop": None,
        "asymmetric_crop": False,
        "method_specific_enhancement": False,
        "ai_generated": False,
    }


def select_first_by_garment(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["garment"]].append(entry)
    selected: List[Dict[str, Any]] = []
    for garment in GARMENTS:
        items = sorted(grouped.get(garment, []), key=lambda item: (
            int(item["rotation"]), str(item["query_id"]), int(item["seed"]), item["asset_id"]
        ))
        if items:
            selected.append(items[0])
    return selected


def sheet_entries(data: Dict[str, Any], asset_by_path: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for item in data["selection"]["items"]:
        asset = asset_by_path[str(item["path"])]
        values.append({
            "asset_id": asset["asset_id"],
            "source_path": str(item["path"]),
            "source_sha256": asset["original_sha256"],
            "method": "SEVEN_METHOD_COMPARISON_SHEET",
            "rotation": item["rotation"],
            "seed": item["seed"],
            "garment": item["garment"],
            "query_id": item["condition"],
            "perturbation": "clean",
            "realized_endpoint": item["garment"],
            "correct_endpoint": item["garment"],
            "figure_eligibility": "REQUIRES_MANUAL_ADJUDICATION",
            "category": item["category"],
        })
    return values


def overview_entries(data: Dict[str, Any], asset_by_path: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    values = []
    for index, path_value in enumerate(data["selection"]["overview_paths"]):
        path = str(path_value)
        asset = asset_by_path[path]
        values.append({
            "asset_id": asset["asset_id"], "source_path": path,
            "source_sha256": asset["original_sha256"], "method": "OVERVIEW",
            "rotation": "MULTIPLE", "seed": "MULTIPLE", "garment": "MULTIPLE",
            "query_id": f"overview_{index:02d}", "perturbation": "clean",
            "realized_endpoint": "MULTIPLE", "correct_endpoint": "MULTIPLE",
            "figure_eligibility": "SUPPLEMENTARY_PARITY_SAFETY_CANDIDATE",
        })
    return values


def build_artifacts(source_repo: Path, source_attempt: Path, output_root: Path, plot_tool: Path) -> Dict[str, Any]:
    gate = check_source(source_repo, source_attempt)
    if gate["status"] != "PASS":
        raise RuntimeError(json.dumps(gate, sort_keys=True))
    inventory = load_json(output_root / "02_pure_endpoint_inventory/pure_endpoint_source_visual_registry.json")
    data = source_data(source_repo, source_attempt)
    asset_by_path = {item["original_path"]: item for item in inventory["assets"]}
    entries = logical_entries(data, asset_by_path)
    source_sheets = sheet_entries(data, asset_by_path)
    overviews = overview_entries(data, asset_by_path)
    plot_source = build_plot_source(data)
    write_json(output_root / "04_metric_plots/metric_plot_source_data.json", plot_source)
    subprocess.run([
        sys.executable, str(plot_tool),
        "--source-data", str(output_root / "04_metric_plots/metric_plot_source_data.json"),
        "--output-dir", str(output_root / "04_metric_plots"),
    ], check=True)
    contact_definitions: List[Tuple[str, List[Dict[str, Any]], str, Any]] = []
    primary_canon = [e for e in entries if e["phase"] == "primary" and e["method"] == "CanonDressGS-Endpoint"]
    contact_definitions.append((
        "pure_endpoint_clean_all_garments", primary_canon,
        "ALL_4_ROTATIONS_X_5_QUERIES_X_3_SEEDS", 60,
    ))
    equivalence = [e for e in entries if e["phase"] == "primary" and e["method"] in {
        "CanonDressGS-Endpoint", "Reference Classifier Lookup", "Nearest-Centroid Lookup", "Outfit-ID Oracle"
    } and e["rotation"] == 0 and e["seed"] == 0]
    contact_definitions.append((
        "pure_endpoint_methods_clean_equivalence", equivalence,
        "ROTATION_0_SEED_0_ALL_GARMENTS_ALL_FOUR_EQUIVALENT_METHODS", 240,
    ))
    perturb_overview = [e for e in entries if e["phase"] == "perturbation" and e["method"] == "CanonDressGS-Endpoint"
                        and e["rotation"] == 0 and e["seed"] == 0]
    contact_definitions.append((
        "pure_endpoint_perturbation_overview", perturb_overview,
        "ROTATION_0_SEED_0_ALL_GARMENTS_ALL_REGISTERED_PERTURBATIONS", 360,
    ))
    mild_failures = [e for e in entries if e["phase"] == "perturbation"
                     and e["method"] == "CanonDressGS-Endpoint" and e["perturbation"] == "mild_blur"
                     and e["endpoint_exact_match"] is False]
    contact_definitions.append((
        "pure_endpoint_mild_blur_failures", mild_failures,
        "ALL_REGISTERED_MILD_BLUR_ENDPOINT_FLIPS", 60,
    ))
    single_reference = [e for e in entries if e["phase"] == "perturbation"
                        and e["method"] == "CanonDressGS-Endpoint" and e["perturbation"] == "single_reference"]
    contact_definitions.append((
        "pure_endpoint_single_reference", single_reference,
        "ALL_REGISTERED_SINGLE_REFERENCE_CASES", 60,
    ))
    base_unique: Dict[int, Dict[str, Any]] = {}
    for e, logical in zip(entries, data["logical"]):
        if e["phase"] == "primary" and e["method"] == "Base Avatar":
            base_unique.setdefault(int(logical["physical_index"]), e)
    dropout_base = sorted(base_unique.values(), key=lambda e: (GARMENTS.index(e["garment"]), e["query_id"]))
    contact_definitions.append((
        "pure_endpoint_dropout_abstention", dropout_base,
        "ALL_UNIQUE_BASE_AVATAR_OUTPUTS_REFERENCED_BY_80_OF_80_DROPOUT_AUDIT", 80,
    ))
    raw_vs = [e for e in entries if e["phase"] == "primary" and e["method"] in {
        "Linear Coefficient Predictor", "CanonDressGS-Endpoint"
    } and e["rotation"] == 0 and e["seed"] == 0]
    contact_definitions.append((
        "pure_endpoint_raw_vs_snapped", raw_vs,
        "ROTATION_0_SEED_0_ALL_GARMENTS_BOTH_REALIZATIONS", 120,
    ))
    contact_definitions.append((
        "pure_endpoint_identity_safety", overviews,
        "ALL_SIX_SEALED_OVERVIEWS", 120,
    ))
    transforms: List[Dict[str, Any]] = []
    contact_registry: List[Dict[str, Any]] = []
    for name, selected, rule, denominator in contact_definitions:
        selected = sorted(selected, key=lambda e: (
            GARMENTS.index(e["garment"]) if e["garment"] in GARMENTS else 99,
            str(e["perturbation"]), int(e["rotation"]) if isinstance(e["rotation"], int) else 99,
            str(e["query_id"]), int(e["seed"]) if isinstance(e["seed"], int) else 99,
            str(e["method"]), e["asset_id"],
        ))
        path = output_root / "03_contact_sheets" / f"{name}.png"
        transform = build_sheet(name, selected, path, selection_rule=rule, denominator=denominator)
        transforms.append(transform)
        contact_registry.append({
            "sheet_id": name,
            "path": str(path),
            "sha256": transform["output_sha256"],
            "source_asset_ids": transform["input_asset_ids"],
            "selection_rule": rule,
            "denominator": denominator,
            "status": "CONTACT_SHEET_FOR_MANUAL_ADJUDICATION",
        })
    main_sheets = [e for e in source_sheets if e["category"] == "main"]
    main_sheets.sort(key=lambda e: (GARMENTS.index(e["garment"]), int(e["rotation"]), e["query_id"], int(e["seed"])))
    teaser_path = output_root / "05_candidate_panels/fig01_teaser/contact_sheet_for_manual_adjudication.png"
    teaser_transform = build_sheet(
        "FIGURE1_ENDPOINT_ONLY_TEASER_CANDIDATE_V1", main_sheets, teaser_path,
        columns=3, image_box=(640, 110),
        selection_rule="ALL_60_MAIN_SHEETS_MANIFEST_DOES_NOT_UNIQUELY_SELECT_FIVE_TEASER_QUERIES",
        denominator=60,
    )
    transforms.append(teaser_transform)
    plot_root = output_root / "04_metric_plots"
    fig5_plots = [plot_root / f"{name}.png" for name in (
        "hard_lookup_agreement_clean", "hard_lookup_error_overlap_perturbation",
        "endpoint_flip_by_perturbation", "raw_vs_realized_coefficient_error",
    )]
    fig5_path = output_root / "05_candidate_panels/fig05_hard_lookup/pure_endpoint_hard_lookup_v1.png"
    transforms.append(compose_plots("FIGURE5_HARD_LOOKUP_RELATION_CANDIDATE_V1", fig5_plots, fig5_path))
    mild_selected = {entry["garment"]: entry for entry in select_first_by_garment(mild_failures)}
    mild_first: List[Dict[str, Any]] = []
    for garment in GARMENTS:
        if garment in mild_selected:
            mild_first.append(mild_selected[garment])
        else:
            mild_first.append({
                "asset_id": "NO_SOURCE_VISUAL",
                "source_path": None,
                "source_sha256": "",
                "method": "CanonDressGS-Endpoint",
                "rotation": "N/A",
                "seed": "N/A",
                "garment": garment,
                "query_id": "NO_REGISTERED_FLIP",
                "perturbation": "mild_blur",
                "realized_endpoint": "N/A",
                "correct_endpoint": garment,
                "figure_eligibility": "NO_FLIP_FOR_THIS_GARMENT",
                "placeholder_text": "NO_FLIP_FOR_THIS_GARMENT",
            })
    single_first = select_first_by_garment(single_reference)
    dropout_first = [dict(entry) for entry in select_first_by_garment(dropout_base)]
    for entry in dropout_first:
        entry["source_render_variant"] = entry["perturbation"]
        entry["perturbation"] = "complete_dropout"
        entry["predicted_endpoint"] = "ABSTAIN"
        entry["realized_endpoint"] = "Base Avatar"
        entry["correct_endpoint"] = "SAFE_ABSTAIN_TO_BASE_AVATAR"
    for entry in mild_first:
        entry["selection_group"] = "MILD_BLUR_FIRST_FLIP"
    for entry in single_first:
        entry["selection_group"] = "SINGLE_REFERENCE_FIRST_REGISTERED"
    for entry in dropout_first:
        entry["selection_group"] = "DROPOUT_SAFE_BASE_AVATAR"
    fig6_entries = mild_first + single_first + dropout_first
    fig6_path = output_root / "05_candidate_panels/fig06_limitations/pure_endpoint_perturbation_v1.png"
    fig6_transform = build_sheet(
        "FIGURE6_PURE_ENDPOINT_PERTURBATION_LIMITATION_V1", fig6_entries, fig6_path,
        columns=5, selection_rule="PER_GARMENT_REGISTERED_LEXICAL_SELECTION", denominator="60/60/80",
    )
    transforms.append(fig6_transform)
    supp_plots = [plot_root / f"{name}.png" for name in (
        "clean_method_top1", "endpoint_exact_match", "identity_and_safety_summary",
        "render_equivalence_and_cache_summary",
    )]
    supp_path = output_root / "05_candidate_panels/supplementary/pure_endpoint_parity_safety_v1.png"
    transforms.append(compose_plots("SUPPLEMENTARY_PURE_ENDPOINT_PARITY_AND_SAFETY_V1", supp_plots, supp_path))
    metric_manifest = load_json(plot_root / "metric_plot_manifest.json")
    for plot in metric_manifest["plots"]:
        transforms.append({
            "transform_id": f"PE-PLOT-{plot['plot_id'].upper().replace('_', '-')}",
            "operation": "DETERMINISTIC_MATPLOTLIB_FROM_SEALED_STRUCTURED_JSON",
            "parameters": {"source_files": plot["source_files"]},
            "input_sha256": [],
            "outputs": plot["outputs"],
            "output_root": str(plot_root),
            "crop": None,
            "asymmetric_crop": False,
            "method_specific_enhancement": False,
            "ai_generated": False,
            "claim_status": plot["claim_status"],
        })
    selected_asset_ids = sorted({
        asset_id for transform in transforms for asset_id in transform.get("input_asset_ids", []) if asset_id
    })
    candidates = [
        {
            "candidate_id": "FIGURE1_ENDPOINT_ONLY_TEASER_CANDIDATE_V1",
            "figure_id": "FIG01_TEASER",
            "status": "REQUIRES_MANUAL_ADJUDICATION",
            "substatus": "PURE_ENDPOINT_ENDPOINT_ONLY_CANDIDATE",
            "panel_path": str(teaser_path),
            "source_asset_ids": teaser_transform["input_asset_ids"],
            "selection_rule": "ALL_60_MAIN_SHEETS_PENDING_MANUAL_FIVE_QUERY_SELECTION",
            "claim_boundary": "FIXED_SUBJECT02_CLOSED_FIVE_GARMENT_SEEN_REFERENCES",
        },
        {
            "candidate_id": "FIGURE5_HARD_LOOKUP_RELATION_CANDIDATE_V1",
            "figure_id": "FIG05_HARD_LOOKUP_ENDPOINT",
            "status": "REQUIRES_MANUAL_ADJUDICATION",
            "substatus": "SEALED_PURE_ENDPOINT_HARD_LOOKUP_EVIDENCE",
            "panel_path": str(fig5_path),
            "source_asset_ids": [],
            "selection_rule": "SEALED_AGGREGATE_DENOMINATORS_AND_ONE_EQUIVALENCE_STRATUM",
            "claim_boundary": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY",
        },
        {
            "candidate_id": "FIGURE6_PURE_ENDPOINT_PERTURBATION_LIMITATION_V1",
            "figure_id": "FIG06_LIMITATIONS",
            "status": "HISTORICAL_LIMITATION_ONLY",
            "substatus": "PURE_ENDPOINT_LIMITATION_PANEL_AVAILABLE",
            "panel_path": str(fig6_path),
            "source_asset_ids": fig6_transform["input_asset_ids"],
            "selection_rule": "FIRST_FLIP_OR_FIRST_REGISTERED_PER_GARMENT_LEXICAL",
            "claim_boundary": "PERTURBATION_SENSITIVITY_DESCRIPTIVE_ONLY",
        },
        {
            "candidate_id": "SUPPLEMENTARY_PURE_ENDPOINT_PARITY_AND_SAFETY_V1",
            "figure_id": "SUPP_PURE_ENDPOINT_PARITY_SAFETY",
            "status": "REQUIRES_MANUAL_ADJUDICATION",
            "substatus": "ENGINEERING_PARITY_AND_SAFETY",
            "panel_path": str(supp_path),
            "source_asset_ids": [],
            "selection_rule": "ALL_SEALED_AGGREGATE_CHECKS",
            "claim_boundary": "SAME_ENDPOINT_PARITY_NOT_IMAGE_QUALITY_SUPERIORITY",
        },
    ]
    transform_registry = {
        "schema_version": "paper_figure_transform_registry_pure_endpoint_refresh.v1",
        "task_id": TASK_ID,
        "transforms": sorted(transforms, key=lambda item: item["transform_id"]),
    }
    candidate_registry = {
        "schema_version": "paper_figure_candidate_registry_pure_endpoint_refresh.v1",
        "task_id": TASK_ID,
        "candidates": candidates,
        "selected_source_asset_count": len(selected_asset_ids),
        "selected_source_asset_ids": selected_asset_ids,
        "paper_final": 0,
    }
    claim_registry = {
        "schema_version": "paper_figure_claim_boundary_pure_endpoint_refresh.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "allowed_claims": [
            "Reliable reference-controlled selection of seen garment endpoints in a fixed closed wardrobe.",
            "All clean endpoint methods select the same endpoint and are functionally equivalent.",
            "Perturbation differences are descriptive under registered denominators.",
            "Closed-bank success uses nearest-endpoint realization despite inaccurate raw coefficients.",
        ],
        "forbidden_claims": [
            "outperforms hard lookup", "superior", "significantly better", "robust to perturbations",
            "continuous control", "precise coefficient regression", "unseen garment",
            "strict novel view", "generalizable garment space",
        ],
        "paper_final": 0,
    }
    plot_registry = {
        "schema_version": "paper_figure_pure_endpoint_plot_registry.v1",
        "plot_count": len(metric_manifest["plots"]),
        "plots": metric_manifest["plots"],
    }
    contact_output_registry = {
        "schema_version": "paper_figure_pure_endpoint_contact_sheet_registry.v1",
        "contact_sheet_count": len(contact_registry),
        "contact_sheets": contact_registry,
    }
    source_registry = {
        "schema_version": "paper_figure_pure_endpoint_source_data_registry.v1",
        "source_branch": RESULT_BRANCH,
        "source_head": RESULT_HEAD,
        "execution_head": EXECUTION_HEAD,
        "source_files": [
            {"path": item, "sha256": sha256_file(source_repo / item)} for item in SOURCE_REPO_FILES
        ] + [
            {"path": item, "sha256": sha256_file(source_attempt / item)} for item in SOURCE_OUTPUT_FILES
        ],
        "plot_source_path": str(output_root / "04_metric_plots/metric_plot_source_data.json"),
        "selection_manifest_rule": data["selection"]["selection_rule"],
    }
    write_json(output_root / "06_provenance/pure_endpoint_transform_registry.json", transform_registry)
    write_json(output_root / "06_provenance/pure_endpoint_candidate_registry.json", candidate_registry)
    write_json(output_root / "06_provenance/pure_endpoint_source_data_registry.json", source_registry)
    write_json(output_root / "06_provenance/pure_endpoint_plot_registry.json", plot_registry)
    write_json(output_root / "06_provenance/pure_endpoint_contact_sheet_registry.json", contact_output_registry)
    write_json(output_root / "07_claim_audit/claim_audit.json", claim_registry)
    summary = {
        "schema_version": "paper_figure_pure_endpoint_artifact_summary.v1",
        "task_id": TASK_ID,
        "source_visual_count": inventory["statistics"]["source_visual_count"],
        "source_unique_sha": inventory["statistics"]["source_unique_sha"],
        "source_duplicate_records": inventory["statistics"]["source_duplicate_records"],
        "selected_source_count": len(selected_asset_ids),
        "copied_candidate_count": len(candidates),
        "contact_sheet_count": len(contact_registry),
        "plot_count": len(metric_manifest["plots"]),
        "duplicate_avoided_bytes": inventory["statistics"]["duplicate_avoided_bytes"],
        "mild_blur_flip_count": len(mild_failures),
        "mild_blur_denominator": 60,
        "single_reference_flip_count": sum(1 for e in single_reference if e["endpoint_exact_match"] is False),
        "single_reference_denominator": 60,
        "dropout_safe_abstention_count": sum(
            is_safe_dropout_abstention(row) for row in data["dropout"]["records"]
        ),
        "dropout_denominator": data["dropout"]["record_count"],
        "paper_final": 0,
    }
    write_json(output_root / "08_final_verification/artifact_summary.json", summary)
    return summary


def verify_output(output_root: Path) -> Dict[str, Any]:
    required = [output_root / dirname for dirname in OUTPUT_DIRS]
    failures = [f"MISSING_DIRECTORY:{path}" for path in required if not path.is_dir()]
    inventory = load_json(output_root / "02_pure_endpoint_inventory/pure_endpoint_source_visual_registry.json")
    transform = load_json(output_root / "06_provenance/pure_endpoint_transform_registry.json")
    candidate = load_json(output_root / "06_provenance/pure_endpoint_candidate_registry.json")
    for item in transform["transforms"]:
        if item.get("asymmetric_crop") or item.get("method_specific_enhancement") or item.get("ai_generated"):
            failures.append(f"UNSAFE_TRANSFORM:{item['transform_id']}")
        if "output_path" in item:
            path = Path(item["output_path"])
            if not path.is_file() or sha256_file(path) != item["output_sha256"]:
                failures.append(f"TRANSFORM_SHA_MISMATCH:{item['transform_id']}")
        elif "outputs" in item:
            for suffix, expected in item["outputs"].items():
                extension = "source.json" if suffix == "source_json" else suffix
                plot_id = item["transform_id"].removeprefix("PE-PLOT-").lower().replace("-", "_")
                path = Path(item["output_root"]) / f"{plot_id}.{extension}"
                if not path.is_file() or sha256_file(path) != expected:
                    failures.append(f"PLOT_SHA_MISMATCH:{item['transform_id']}:{suffix}")
    if inventory["statistics"]["source_visual_count"] != 3046:
        failures.append("SOURCE_VISUAL_COUNT_MISMATCH")
    if len(candidate["candidates"]) != 4:
        failures.append("CANDIDATE_COUNT_MISMATCH")
    report = {
        "schema_version": "paper_figure_pure_endpoint_refresh_verification.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source_assets_checked": len(inventory["assets"]),
        "transforms_checked": len(transform["transforms"]),
        "candidates_checked": len(candidate["candidates"]),
        "headroom_active_output_files_read": 0,
        "avatarrex_media_exports": 0,
        "paper_final": 0,
    }
    write_json(output_root / "08_final_verification/verification.json", report)
    if failures:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    gate = check_source(args.source_repo, args.source_attempt)
    if args.dry_run:
        print(json.dumps({
            "status": "VALIDATED_NOT_EXECUTED" if gate["status"] == "PASS" else "BLOCKED",
            "stage": args.stage,
            "gate": gate,
            "output_root": str(args.output_root),
            "writes_headroom": False,
            "reads_headroom": False,
        }, indent=2, sort_keys=True))
        return 0 if gate["status"] == "PASS" else 1
    results: Dict[str, Any] = {"stage": args.stage}
    if args.stage in {"inventory", "all"}:
        results["inventory"] = run_inventory(args.source_repo, args.source_attempt, args.output_root)
    if args.stage in {"artifacts", "all"}:
        if not args.plot_tool or not args.plot_tool.is_file():
            raise SystemExit("--plot-tool is required for artifact generation")
        results["artifacts"] = build_artifacts(
            args.source_repo, args.source_attempt, args.output_root, args.plot_tool
        )
    if args.stage in {"verify", "all"}:
        results["verification"] = verify_output(args.output_root)
    log_path = args.output_root / f"08_final_verification/refresh_{args.stage}_log.json"
    write_json(log_path, {"schema_version": "paper_figure_pure_endpoint_refresh_log.v1", **results})
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
