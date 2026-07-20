from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_SEEN = ["O01", "O02", "O03", "O04", "O08"]
EXPECTED_VIEWS = ["cond_000000", "cond_000318", "cond_000017", "cond_000347"]
EXPECTED_SEEDS = [0, 1, 2]
OUTPUT_TEMPLATE = "${CANONDRESSGS_OUTPUT_ROOT}/AAAI27-SEEN-OUTFIT-PAPER/{experiment_id}/seed_{seed}/attempt_{NNN}"


def load_documents(
    config_path: Path,
    manifest_path: Path,
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    return config, manifest, registry


def audit_protocol(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    claim_matrix_text: str | None = None,
    table_figure_plan_text: str | None = None,
) -> dict[str, Any]:
    experiments = list(registry["experiments"])
    executable = [item for item in experiments if item.get("executable") is True]
    historical = [item for item in experiments if item.get("executable") is False]
    not_run = [item for item in executable if item.get("status") == "NOT_RUN"]
    errors: list[str] = []

    if len(experiments) != 55:
        errors.append("REGISTRY_PROTOCOL_COUNT_MISMATCH")
    if len(executable) != 51 or len(not_run) != 51 or len(historical) != 4:
        errors.append("REGISTRY_PROTOCOL_COUNT_MISMATCH")
    if any(item.get("status") != "HISTORICAL_EVIDENCE" for item in historical):
        errors.append("HISTORICAL_STATUS_INVALID")
    if any(item.get("baseline_or_ablation") != "historical_diagnostic" for item in historical):
        errors.append("A8_SCOPE_INVALID")
    if any(item.get("evidence_class") != "HISTORICAL_EVIDENCE" for item in historical):
        errors.append("HISTORICAL_EVIDENCE_CLASS_INVALID")

    trainable = [item for item in executable if item.get("seed") is not None]
    seeds = sorted({item.get("seed") for item in trainable})
    if seeds != EXPECTED_SEEDS or config["training"]["seeds"] != EXPECTED_SEEDS:
        errors.append("SEED_PROTOCOL_MISMATCH")
    if config["training"]["steps"] != 300:
        errors.append("TRAINING_STEP_PROTOCOL_MISMATCH")
    if config["data"]["seen_outfits"] != EXPECTED_SEEN:
        errors.append("SEEN_SPLIT_MISMATCH")
    if config["data"]["held_out_diagnostic"] != "O07":
        errors.append("HELD_OUT_SPLIT_MISMATCH")
    if config["data"]["unused_reserve"] != "O06":
        errors.append("RESERVE_SPLIT_MISMATCH")
    if config["data"]["target_views"] != EXPECTED_VIEWS:
        errors.append("VIEW_PROTOCOL_MISMATCH")
    if config["data"]["episode_count"] != 20 or config["data"]["reference_target_overlap"] != 0:
        errors.append("EPISODE_PROTOCOL_MISMATCH")
    if config["basis"]["rank"] != 4:
        errors.append("BASIS_RANK_MISMATCH")
    if config["permissions"]["outfit_id_in_model"] or config["permissions"]["target_forward_input"]:
        errors.append("FORWARD_BOUNDARY_MISMATCH")
    if config["permissions"]["train_on_o07"] or config["permissions"]["substitute_o06_for_o07"]:
        errors.append("SPLIT_PERMISSION_MISMATCH")
    if config["baselines"]["B1"].get("role") != "optimization_upper_bound":
        errors.append("B1_ROLE_MISMATCH")
    if not config["baselines"]["B2"].get("seen_only"):
        errors.append("B2_SCOPE_MISMATCH")
    if config["tables"]["table_4"].get("status_label") != "Held-out diagnostic — FAIL":
        errors.append("O07_TABLE_LABEL_MISMATCH")
    manifest_fingerprint = registry["frozen_asset_manifest_sha256"]
    if config["paths"]["frozen_asset_manifest_sha256"] != manifest_fingerprint:
        errors.append("FROZEN_MANIFEST_REFERENCE_MISMATCH")
    if any(item.get("frozen_asset_fingerprint") != manifest_fingerprint for item in experiments):
        errors.append("EXPERIMENT_ASSET_REFERENCE_MISMATCH")
    if any(item.get("expected_output_root") != OUTPUT_TEMPLATE for item in experiments):
        errors.append("OUTPUT_CONTRACT_MISMATCH")
    if any(item.get("evaluator_version") != registry["evaluator_version"] for item in experiments):
        errors.append("EVALUATOR_VERSION_MISMATCH")
    evaluator_asset = next(item for item in manifest["assets"] if item["asset_id"] == "evaluator_commit")
    if config.get("source_evaluator_commit") != evaluator_asset["fingerprint"]:
        errors.append("SOURCE_EVALUATOR_PROVENANCE_MISMATCH")
    if config.get("unified_evaluator_commit") != registry["evaluator_version"]:
        errors.append("UNIFIED_EVALUATOR_PROVENANCE_MISMATCH")
    if len(manifest["assets"]) != 19:
        errors.append("FROZEN_ASSET_COUNT_MISMATCH")
    configured_metrics = {
        metric for group in ("coefficient", "residual", "render", "reference", "efficiency")
        for metric in config["metrics"][group]
    }
    if any(set(item["required_metrics"]) != configured_metrics for item in experiments):
        errors.append("REQUIRED_METRIC_CONTRACT_MISMATCH")
    if claim_matrix_text is not None:
        if any(f"| C{index} |" not in claim_matrix_text for index in range(1, 7)):
            errors.append("ALLOWED_CLAIM_MATRIX_INCOMPLETE")
        if any(f"| N{index} |" not in claim_matrix_text for index in range(1, 8)):
            errors.append("PROHIBITED_CLAIM_MATRIX_INCOMPLETE")
    if table_figure_plan_text is not None:
        if any(f"## Table {index}" not in table_figure_plan_text for index in range(1, 5)):
            errors.append("TABLE_PLAN_INCOMPLETE")
        if any(f"## Figure {index}" not in table_figure_plan_text for index in range(1, 7)):
            errors.append("FIGURE_PLAN_INCOMPLETE")

    return {
        "task_id": "AAAI27-UNIFIED-PAPER-RUNNER-EVALUATOR-001",
        "protocol_task_id": registry["task_id"],
        "status": "PASS" if not errors else "REGISTRY_PROTOCOL_COUNT_MISMATCH"
        if "REGISTRY_PROTOCOL_COUNT_MISMATCH" in errors else "FAIL",
        "errors": sorted(set(errors)),
        "total_count": len(experiments),
        "executable_count": len(executable),
        "not_run_count": len(not_run),
        "historical_evidence_count": len(historical),
        "baseline_count": len([key for key in config["baselines"] if key.startswith("B")]),
        "ours_count": 1 if "Ours" in config["baselines"] else 0,
        "ablation_count": len(config["ablations"]),
        "seed_count": len(config["training"]["seeds"]),
        "table_count": len(config["tables"]),
        "figure_count": len(config["figures"]),
        "frozen_asset_count": len(manifest["assets"]),
        "evaluator_metric_count": len(configured_metrics),
        "claim_matrix_parsed": claim_matrix_text is not None,
        "table_figure_plan_parsed": table_figure_plan_text is not None,
        "frozen_asset_manifest_sha256": manifest_fingerprint,
        "evaluator_version": registry["evaluator_version"],
        "executable_ids": [item["experiment_id"] for item in executable],
        "historical_ids": [item["experiment_id"] for item in historical],
    }
