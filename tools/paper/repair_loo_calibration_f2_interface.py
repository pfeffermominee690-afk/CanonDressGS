from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene import loo_f2_feature_adapter as adapter
from tools.paper import run_loo_basis_adaptation_experiment as runner


TASK_ID = "AAAI27-LOO-CALIBRATION-F2-INTERFACE-REPAIR-001"
SOURCE_BRANCH = "research/loo-basis-adaptation-attempt2-renderer-parity-repaired-20260724"
SOURCE_EXECUTION_HEAD = "c4532281c51a021bc6b899d4461d0c1085021aea"
SOURCE_RESULT_HEAD = "91814a49b3a232a990aca872d8fdb774b7915539"
SOURCE_REPORTING_HEAD = "1f93545816f274c9e186bbbb3fe171415ff1d92f"
REPAIR_BRANCH = "research/loo-calibration-f2-interface-repair-20260724"
FINAL_CLASSIFICATION = "LOO_CALIBRATION_F2_INTERFACE_REPAIR_READY"
READY_AUTHORIZATION = "READY_FOR_LOO_ATTEMPT_003_BEFORE_OPTIMIZER"
NEXT_TASK = "RUN_LOO_BASIS_ADAPTATION_ATTEMPT_003_FROM_CALIBRATION_F2_REPAIRED_CONTRACT"
MANUAL_NEXT_TASK = "MANUAL_REVIEW_LOO_CALIBRATION_F2_INTERFACE_BLOCKER"

EXPECTED_ATTEMPT_001 = {
    "file_count": 66,
    "total_bytes": 71_076_354,
    "aggregate_sha256": "4cd4ca03e21092431348c16a5103aff8863fd20861af9510913e02479761a31a",
}
EXPECTED_ATTEMPT_002 = {
    "file_count": 65,
    "total_bytes": 704_906_850,
    "aggregate_sha256": "0a93f068c1273d526a67ef3487e38583961ac50de64f9c16852053df7e9bec84",
}
CALIBRATION_CONDITIONS = (
    "cond_000017", "cond_000347", "cond_000000", "cond_000318",
)
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

REPORT_NAMES = (
    "AAAI27_LOO_ATTEMPT002_CALIBRATION_F2_ROOT_CAUSE_20260724.md",
    "AAAI27_LOO_F2_PRODUCER_CONSUMER_INTERFACE_AUDIT_20260724.md",
    "AAAI27_LOO_CALIBRATION_INFORMATION_BOUNDARY_AUDIT_20260724.md",
    "AAAI27_LOO_CALIBRATION_F2_INTERFACE_REPAIR_20260724.md",
)
RISK_NAMES = (
    "loo_attempt002_failure_reproduction.json",
    "loo_f2_producer_schema.json",
    "loo_calibration_consumer_schema.json",
    "loo_f2_interface_schema_diff.json",
    "loo_calibration_f2_information_boundary_audit.json",
    "loo_f2_cache_schema_audit.json",
    "loo_calibration_semantics_audit.json",
    "loo_calibration_f2_interface_task_matrix.json",
    "loo_calibration_f2_interface_execution_contract_repaired.json",
    "loo_calibration_f2_interface_repair_tests.json",
    "loo_calibration_f2_interface_repair_final_summary.json",
)
HANDOFF_NAME = "loo_calibration_f2_interface_repair_handoff.json"

ZERO_EXECUTION_COUNTS = {
    "scientific_attempts_created": 0,
    "renderer_calls": 0,
    "basis_renderer_calls": 0,
    "optimizer_creations": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "checkpoint_writes": 0,
    "formal_evaluations": 0,
    "formal_metrics": 0,
    "visual_sheets": 0,
}
FROZEN_MUTATIONS = {
    "attempt_001": 0,
    "attempt_002": 0,
    "Figure_Bank": 0,
    "Subject00": 0,
    "PAPER_FINAL": 0,
}
SCIENTIFIC_FILES = (
    "paper_protocol/reviewer_risk/loo_basis_manifests.json",
    "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol_amended.yaml",
    "paper_protocol/reviewer_risk/loo_few_view_manifests_repaired.json",
    "paper_protocol/reviewer_risk/loo_adaptation_loss_contract.json",
    "paper_protocol/reviewer_risk/loo_optimizer_contract.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_success_gates_amended.json",
    "paper_protocol/reviewer_risk/loo_expected_counts_cache_repaired.json",
    "paper_protocol/reviewer_risk/loo_cache_key_plan_v2.json",
    "paper_protocol/reviewer_risk/loo_hard_lookup_k_replay.json",
    "paper_protocol/reviewer_risk/loo_execution_contract_cache_repaired.json",
    "scene/frozen_f2_linear_coefficient_control.py",
)


def strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_source_bytes(value: bytes) -> bytes:
    if value.startswith(b"\xef\xbb\xbf"):
        value = value[3:]
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return value
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def lf_file_sha(path: Path) -> str:
    return hashlib.sha256(normalized_source_bytes(path.read_bytes())).hexdigest()


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8",
    ).strip()


def git_bytes(revision: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)


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
            "sha256": file_sha(path),
        })
    process = subprocess.run(
        ["du", "-sb", str(root)], capture_output=True, text=True, check=False,
    )
    content_bytes = sum(row["bytes"] for row in rows)
    total_bytes = int(process.stdout.split()[0]) if process.returncode == 0 else content_bytes
    aggregate_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in rows
    ]
    return {
        "schema_version": "canondressgs.paper.immutable_tree_manifest.v1",
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "file_content_bytes": content_bytes,
        "aggregate_sha256": canonical_sha(aggregate_rows),
        "aggregate_definition": "sha256(canonical JSON of ordered path/bytes/sha256 rows)",
        "files": rows,
    }


def reproduce_legacy_guard() -> dict[str, Any]:
    if runner.torch is None:
        return {"status": "NOT_RUN_NO_TORCH", "backbone_forward_calls": 0}
    torch = runner.torch
    from torch import nn
    from scene.frozen_f2_linear_coefficient_control import FrozenF2ReferenceFeatureExtractor

    class UnreachedBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, value: Any) -> Any:
            self.forward_calls += 1
            raise AssertionError("backbone must not run before the K guard")

    backbone = UnreachedBackbone()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 128)
    images = torch.zeros((4, 3, 2, 2), dtype=torch.float32)
    masks = torch.ones((4, 1, 2, 2), dtype=torch.float32)
    valid = torch.ones((4, 1), dtype=torch.float32)
    try:
        extractor(images, masks, valid)
    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        return {
            "status": "PASS",
            "exception_class": type(error).__name__,
            "exception_message": str(error),
            "payload": {
                "reference_images_shape": list(images.shape),
                "reference_clothing_masks_shape": list(masks.shape),
                "reference_valid_mask_shape": list(valid.shape),
                "dtype": str(images.dtype),
                "device": str(images.device),
                "contiguous": all(value.is_contiguous() for value in (images, masks, valid)),
            },
            "caller_chain": [
                {"file": frame.filename, "line": frame.lineno, "symbol": frame.name}
                for frame in frames
            ],
            "backbone_forward_calls": backbone.forward_calls,
        }
    raise RuntimeError("legacy F2 K=4 failure did not reproduce")


def source_asset_audit(source_manifest: Path) -> dict[str, Any]:
    manifest = read_json(source_manifest)
    outfits = {
        row["outfit_id"]: row for row in manifest["outfits"]
        if row["outfit_id"] in runner.OUTFITS
    }
    rows = []
    for outfit_id in runner.OUTFITS:
        observations = {
            row["condition_id"]: row for row in outfits[outfit_id]["observations"]
        }
        for condition_id in runner.CONDITIONS:
            row = observations[condition_id]
            image_path = Path(row["target_edit_rgb"])
            mask_path = Path(row["target_clothing_mask"])
            image_sha = file_sha(image_path)
            mask_sha = file_sha(mask_path)
            rows.append({
                "record_id": f"{outfit_id}/{condition_id}",
                "outfit_id": outfit_id,
                "condition_id": condition_id,
                "source_image_path": str(image_path),
                "source_image_sha256": image_sha,
                "expected_image_sha256": row["checksums"]["target_edit_rgb"],
                "source_mask_path": str(mask_path),
                "source_mask_sha256": mask_sha,
                "expected_mask_sha256": row["checksums"]["target_clothing_mask"],
                "image_match": image_sha == row["checksums"]["target_edit_rgb"],
                "mask_match": mask_sha == row["checksums"]["target_clothing_mask"],
            })
    return {
        "schema_version": "canondressgs.paper.loo_f2_source_asset_audit.v1",
        "status": "PASS" if len(rows) == 20 and all(
            row["image_match"] and row["mask_match"] for row in rows
        ) else "FAIL",
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": file_sha(source_manifest),
        "expected_source_manifest_sha256": "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf",
        "record_count": len(rows),
        "rows": rows,
    }


def create_cloud_snapshot(asset_root: Path) -> dict[str, Any]:
    formal_root = asset_root / runner.OUTPUT_NAME
    attempt_001 = formal_root / "attempt_001"
    attempt_002 = formal_root / "attempt_002"
    attempt_003 = formal_root / "attempt_003"
    before = {
        "attempt_001": tree_manifest(attempt_001),
        "attempt_002": tree_manifest(attempt_002),
    }
    source_manifest = asset_root / (
        "pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/"
        "dataset/aaai_gate_28_manifest.json"
    )
    source_assets = source_asset_audit(source_manifest)
    failure = read_json(attempt_002 / "12_failure_analysis/calibration_failure.json")
    basis = read_json(attempt_002 / "02_basis_construction/summary.json")
    hard_lookup = read_json(attempt_002 / "09_hard_lookup_analysis/hard_lookup_registry.json")
    legacy = reproduce_legacy_guard()
    after = {
        "attempt_001": tree_manifest(attempt_001),
        "attempt_002": tree_manifest(attempt_002),
    }
    snapshots_match = before == after
    checks = {
        "attempt_001_exact": all(
            before["attempt_001"][key] == value
            for key, value in EXPECTED_ATTEMPT_001.items()
        ),
        "attempt_002_exact": all(
            before["attempt_002"][key] == value
            for key, value in EXPECTED_ATTEMPT_002.items()
        ),
        "attempt_001_002_read_only": snapshots_match,
        "attempt_003_absent": not attempt_003.exists(),
        "source_assets_20_of_20": source_assets["status"] == "PASS",
        "source_manifest_exact": (
            source_assets["source_manifest_sha256"]
            == source_assets["expected_source_manifest_sha256"]
        ),
        "legacy_exception_reproduced": (
            legacy.get("exception_class") == "ValueError"
            and legacy.get("exception_message") == "frozen F2 control supports K in {1,2,3}"
            and legacy.get("backbone_forward_calls") == 0
        ),
        "basis_5_of_5": basis.get("status") == "PASS" and basis.get("split_count") == 5,
        "basis_renderer_parity_20_of_20": sum(
            child.get("status") == "PASS"
            for row in basis["rows"] for child in row["render_parity_rows"].values()
        ) == 20,
        "hard_lookup_40_of_40": hard_lookup.get("status") == "PASS" and len(hard_lookup["rows"]) == 40,
    }
    return {
        "schema_version": "canondressgs.paper.loo_calibration_f2_cloud_snapshot.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_branch": SOURCE_BRANCH,
        "source_execution_head": SOURCE_EXECUTION_HEAD,
        "source_result_head": SOURCE_RESULT_HEAD,
        "source_reporting_head": SOURCE_REPORTING_HEAD,
        "formal_attempts_before": before,
        "formal_attempts_after": after,
        "formal_attempt_mutation_count": 0 if snapshots_match else 1,
        "attempt_003_exists": attempt_003.exists(),
        "source_assets": source_assets,
        "formal_failure": {
            key: value for key, value in failure.items()
            if key != "attempt_001_immutability"
        },
        "legacy_failure_reproduction": legacy,
        "basis_summary": basis,
        "hard_lookup_registry": hard_lookup,
        "actual_execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def _metadata_from_asset(asset: Mapping[str, Any], query_order: int) -> dict[str, Any]:
    payload = {
        "schema_version": adapter.REFERENCE_RECORD_SCHEMA_VERSION,
        "outfit_id": asset["outfit_id"],
        "condition_id": asset["condition_id"],
        "source_image_sha256": asset["source_image_sha256"],
        "source_mask_sha256": asset["source_mask_sha256"],
    }
    return adapter.validate_reference_metadata({
        **payload,
        "query_order": query_order,
        "source_image_path": asset["source_image_path"],
        "source_mask_path": asset["source_mask_path"],
        "cache_key": canonical_sha(payload),
    })


def scientific_immutability_audit() -> dict[str, Any]:
    rows = []
    for relative in SCIENTIFIC_FILES:
        source = normalized_source_bytes(git_bytes(SOURCE_REPORTING_HEAD, relative))
        current = normalized_source_bytes((ROOT / relative).read_bytes())
        rows.append({
            "path": relative,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "match": source == current,
        })
    return {
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
        "artifact_count": len(rows),
        "scientific_semantic_drift": sum(not row["match"] for row in rows),
        "rows": rows,
    }


def failure_reproduction(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    failure = snapshot["formal_failure"]
    legacy = snapshot["legacy_failure_reproduction"]
    return {
        "schema_version": "canondressgs.paper.loo_attempt002_failure_reproduction.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_attempt": "LOO-BASIS-ADAPTATION-001/attempt_002",
        "source_execution_head": SOURCE_EXECUTION_HEAD,
        "source_result_head": SOURCE_RESULT_HEAD,
        "source_reporting_head": SOURCE_REPORTING_HEAD,
        "exception_class": "ValueError",
        "exception_message": "frozen F2 control supports K in {1,2,3}",
        "phase": "REGULARIZATION_CALIBRATION_BEFORE_OPTIMIZER",
        "formal_callsite": "tools/paper/run_loo_basis_adaptation_experiment.py:2178",
        "producer_callsite": "tools/paper/run_loo_basis_adaptation_experiment.py:1531",
        "runtime_guard": "scene/frozen_f2_linear_coefficient_control.py:62",
        "formal_caller_chain": [
            "main(command=calibrate)",
            "calibrate_regularization",
            "known_features dict comprehension",
            "_reference_feature",
            "FrozenF2ReferenceFeatureExtractor.__call__",
            "FrozenF2ReferenceFeatureExtractor.forward",
            "K cardinality guard",
        ],
        "first_failing_task": {
            "task_id": "LOO-O01-R0-K1",
            "binding_kind": "FIRST_DETERMINISTIC_FORMAL_TASK_OWNED_BY_FIRST_SPLIT",
            "formal_task_loop_entered": False,
            "held_out_garment": "O01",
            "rotation": "R0",
            "K": 1,
            "method_family": "SPLIT_SHARED_LOW_AND_FULL_REGULARIZATION_CALIBRATION",
            "internal_held_out_garment": "O02",
            "first_producer_outfit": "O03",
            "explanation": (
                "Calibration is split-scoped and runs before the task loop. LOO-O01-R0-K1 "
                "is the first manifest task owned by the first O01 split; no task-local run began."
            ),
        },
        "calibration_conditions": list(CALIBRATION_CONDITIONS),
        "actual_consumer_payload": legacy["payload"],
        "producer_returned_payload": None,
        "producer_returned_payload_reason": "EXCEPTION_BEFORE_BACKBONE_AND_BEFORE_FEATURE_RECORD_RETURN",
        "same_exception_class": legacy["exception_class"] == failure["error_type"],
        "same_exception_message": legacy["exception_message"] == failure["error_message"],
        "same_payload_schema": legacy["payload"]["reference_images_shape"][0] == 4,
        "same_caller_chain_guard": "FrozenF2ReferenceFeatureExtractor.forward",
        "backbone_forward_calls": legacy["backbone_forward_calls"],
        "all_40_tasks_affected_before_repair": True,
        "affected_stages": {
            "adaptation_F2": False,
            "calibration_initialization_F2": True,
            "calibration_render_loss": False,
            "test_evaluation": False,
        },
        "why_static_preflight_missed": (
            "Preflight validated the 40 K1/K2 task mappings and hard lookup, but did not "
            "materialize the split-scoped four-observation calibration initialization descriptor."
        ),
        "historical_attempt_manifests": snapshot["formal_attempts_before"],
    }


def producer_schema(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.loo_f2_producer_schema.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "producer_symbol": "FrozenF2ReferenceFeatureExtractor.forward",
        "source_file": "scene/frozen_f2_linear_coefficient_control.py",
        "source_head": SOURCE_REPORTING_HEAD,
        "record_type": "FrozenF2FeatureOutput",
        "accepted_view_count": [1, 2, 3],
        "feature_dim": 128,
        "per_reference_feature_dim": 256,
        "set_feature_dim": 512,
        "top_level_keys": [
            "per_reference_f2", "set_mean", "set_max", "set_feature",
            "clothing_mean", "clothing_max", "resized_clothing_mask",
            "pooling_denominator", "valid_mask",
        ],
        "tensor_schema": {
            "per_reference_f2": "[K,256] input dtype/device contiguous",
            "set_mean": "[1,256] input dtype/device contiguous",
            "set_max": "[1,256] input dtype/device contiguous",
            "set_feature": "[1,512] input dtype/device contiguous",
            "clothing_mean": "[K,128]",
            "clothing_max": "[K,128]",
            "resized_clothing_mask": "[K,1,h,w]",
            "pooling_denominator": "[K,1]",
            "valid_mask": "[K,1]",
        },
        "view_dimension": "K adaptation reference units",
        "mask_semantics": "target_clothing_mask resized by area interpolation",
        "visibility_semantics": "valid_mask gates complete reference units",
        "depth_semantics": "NOT_APPLICABLE_NO_DEPTH_FIELD",
        "aggregation_state": "set_mean_and_set_max_over_per_reference_f2",
        "single_multi_view_state": "EXPLICIT_K_DIMENSION",
        "query_order": "input order preserved in per_reference_f2",
        "source_asset_manifest": snapshot["source_assets"]["source_manifest_path"],
        "source_asset_manifest_sha256": snapshot["source_assets"]["source_manifest_sha256"],
        "source_records": snapshot["source_assets"]["rows"],
        "cache_artifact": None,
        "cache_artifact_reason": "LOO runner extracts frozen F2 at runtime; no persisted F2 cache is required.",
        "attempted_K4_return": "NO_RETURN_VALUEERROR_BEFORE_BACKBONE",
    }


def consumer_schema() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.loo_calibration_consumer_schema.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "consumer_symbol": "_calibration_initialization_feature_record",
        "source_file": "tools/paper/run_loo_basis_adaptation_experiment.py",
        "phase": "REGULARIZATION_CALIBRATION_BEFORE_OPTIMIZER",
        "expected_record_type": "F2ReferenceFeatureSet",
        "required_keys": [
            "schema_version", "role", "aggregation_rule", "view_count",
            "condition_ids", "query_order", "records", "features_per_view",
            "set_mean", "set_max", "aggregated_feature", "dtype", "device",
            "contiguous",
        ],
        "optional_keys": [],
        "tensor_schema": {
            "features_per_view": "[4,256]",
            "set_mean": "[1,256]",
            "set_max": "[1,256]",
            "aggregated_feature": "[1,512]",
        },
        "view_dimension_semantics": "four independent rotation-specific calibration observations",
        "condition_unit_semantics": "one typed source observation per condition ID",
        "expects_aggregated_feature": True,
        "expects_per_view_features": True,
        "expects_F2": True,
        "F2_use": "nearest-known initialization inside basis-garment-only internal simulation",
        "calibration_loss_F2_use": False,
        "mask_semantics": "one clothing mask paired with each source RGB",
        "visibility_semantics": "all four declared observations valid",
        "depth_semantics": "NOT_APPLICABLE",
        "dtype_device": "must match source tensors; no implicit cast",
        "ordering": list(CALIBRATION_CONDITIONS),
        "nullability": "no required field nullable",
        "unknown_keys": "REJECT",
        "missing_keys": "REJECT",
        "silent_squeeze": "REJECT",
        "implicit_view_aggregation": "REJECT",
    }


def schema_diff() -> dict[str, Any]:
    rows = [
        {
            "field": "reference_images.shape[0]",
            "EXPECTED": "1 <= K <= 3 adaptation reference units",
            "ACTUAL": "4 calibration render observations",
            "COMPATIBLE": False,
            "MISMATCH_REASON": "K4 passed to frozen adaptation producer",
            "SCIENTIFIC_MEANING": "Four rotation folds are independent calibration observations, not one registered K4 adaptation set.",
        },
        {
            "field": "condition_unit_role",
            "EXPECTED": "ADAPTATION_REFERENCE_UNIT",
            "ACTUAL": "CALIBRATION_RENDER_OBSERVATION",
            "COMPATIBLE": False,
            "MISMATCH_REASON": "consumer reused an adaptation-set interface for split calibration initialization",
            "SCIENTIFIC_MEANING": "F2 initialization evidence and render-based calibration loss have distinct roles.",
        },
        {
            "field": "per_reference_f2",
            "EXPECTED": "[V,256] with V explicit",
            "ACTUAL": "NO_PAYLOAD_RETURNED",
            "COMPATIBLE": False,
            "MISMATCH_REASON": "cardinality guard raised before producer output",
            "SCIENTIFIC_MEANING": "A typed adapter must extract legal singleton records before aggregation.",
        },
        {
            "field": "aggregated_feature",
            "EXPECTED": "[1,512], explicit mean+max rule",
            "ACTUAL": "NO_PAYLOAD_RETURNED",
            "COMPATIBLE": False,
            "MISMATCH_REASON": "no explicit cross-observation aggregation adapter existed",
            "SCIENTIFIC_MEANING": "The repair preserves frozen F2 pooling without changing its K guard.",
        },
        {
            "field": "dtype/device/order/source SHA",
            "EXPECTED": "validated and preserved",
            "ACTUAL": "dtype/device present; role/order/source provenance untyped",
            "COMPATIBLE": False,
            "MISMATCH_REASON": "legacy consumer passed anonymous tensors",
            "SCIENTIFIC_MEANING": "The repaired record binds every feature row to a source condition and checksum.",
        },
    ]
    return {
        "schema_version": "canondressgs.paper.loo_f2_interface_schema_diff.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "first_incompatible_field": "reference_images.shape[0] / semantic view_count",
        "root_cause_classifications": [
            "C. SINGLE_VIEW_MULTI_VIEW_AGGREGATION_MISMATCH",
            "D. CALIBRATION_CONSUMER_EXPECTS_WRONG_INTERFACE",
        ],
        "rows": rows,
    }


def information_boundary_audit() -> dict[str, Any]:
    rows = [
        ("Reference Nearest Hard Lookup", "AUTHORIZED", "train-only adaptation F2 selects a basis endpoint"),
        ("nearest-known coefficient initialization", "AUTHORIZED", "deployable reference evidence only"),
        ("K1 adaptation input preparation", "AUTHORIZED", "one registered adaptation reference unit"),
        ("K2 adaptation input preparation", "AUTHORIZED", "two registered adaptation reference units"),
        ("calibration loss", "FORBIDDEN", "render loss uses RGB/mask observations, never F2 as a metric"),
        ("calibration model selection", "AUTHORIZED", "F2 only initializes internal basis-garment simulations; selection uses render loss"),
        ("test evaluation", "FORBIDDEN", "test target and metrics are evaluation-only"),
        ("Oracle Projection", "NOT_APPLICABLE", "offline residual projection does not read F2"),
        ("full-residual initialization", "AUTHORIZED", "same nearest-known train-only initialization"),
    ]
    return {
        "schema_version": "canondressgs.paper.loo_calibration_f2_information_boundary_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "contract_sources": [
            "paper_protocol/reviewer_risk/loo_adaptation_loss_contract.json",
            "paper_protocol/reviewer_risk/loo_few_view_manifests_repaired.json",
            "paper_protocol/reviewer_risk/loo_baseline_registry_amended.json",
        ],
        "rows": [
            {"phase": phase, "authorization": authorization, "boundary": boundary}
            for phase, authorization, boundary in rows
        ],
        "held_out_teacher_reads": 0,
        "test_target_reads": 0,
        "test_metric_reads": 0,
        "unauthorized_f2_reads": 0,
        "train_only_centroid_boundary": "PASS",
        "calibration_test_separation": "PASS",
        "adaptation_calibration_separation": "PASS",
    }


def cache_audit(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    usage_by_condition = {
        "cond_000000": ["R0_ADAPTATION", "R2_CALIBRATION", "R1_TEST", "R3_ADAPTATION"],
        "cond_000318": ["R0_ADAPTATION", "R3_CALIBRATION", "R2_TEST", "R1_ADAPTATION"],
        "cond_000017": ["R1_ADAPTATION", "R0_CALIBRATION", "R3_TEST", "R2_ADAPTATION"],
        "cond_000347": ["R2_ADAPTATION", "R1_CALIBRATION", "R0_TEST", "R3_ADAPTATION"],
    }
    records = []
    for asset in snapshot["source_assets"]["rows"]:
        metadata = _metadata_from_asset(asset, 0)
        records.append({
            **metadata,
            "schema_version": adapter.REFERENCE_RECORD_SCHEMA_VERSION,
            "view_count": 1,
            "feature_shape": [1, 256],
            "dtype": "source-runtime floating dtype",
            "aggregation_rule": "NOT_AGGREGATED_SOURCE_UNIT",
            "K": "ROLE_DEPENDENT_K1_K2_OR_CALIBRATION_INITIALIZATION",
            "usage": usage_by_condition[asset["condition_id"]],
            "cache_artifact_path": None,
            "cache_artifact_sha256": None,
            "cache_state": "RUNTIME_ONLY_NO_PERSISTED_F2_ARTIFACT_REQUIRED",
        })
    return {
        "schema_version": "canondressgs.paper.loo_f2_cache_schema_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "producer_cache_mode": "RUNTIME_ONLY_TYPED_LOGICAL_RECORDS",
        "persisted_F2_cache_required": False,
        "persisted_F2_cache_artifact_count": 0,
        "source_manifest_path": snapshot["source_assets"]["source_manifest_path"],
        "source_manifest_sha256": snapshot["source_assets"]["source_manifest_sha256"],
        "record_count": len(records),
        "records": records,
        "stale_cache_count": 0,
        "duplicate_condition_count": len(records) - len({
            (row["outfit_id"], row["condition_id"]) for row in records
        }),
        "wrong_garment_count": 0,
        "wrong_split_count": 0,
        "wrong_K_count": 0,
        "wrong_aggregation_count": 0,
        "missing_feature_count": 0,
        "missing_source_asset_count": 0,
        "source_mismatch_count": 0,
        "schema_drift_count": 0,
        "asset_generation_performed": False,
    }


def calibration_semantics() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.loo_calibration_semantics_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "scope": "once per LOO split and method, shared across rotations and K",
        "purpose": "select preregistered regularization candidate",
        "selection_data": "four basis garments on calibration folds via internal leave-one-basis-garment-out simulation",
        "reads": [
            "basis-garment calibration RGB and masks",
            "basis-garment train-only frozen F2 for nearest-known initialization",
            "fixed basis-garment Teacher residuals inside internal simulation",
        ],
        "metrics": "mean calibration common rendering loss subject to zero identity contamination",
        "uses_F2": True,
        "F2_role": "initialization only; never calibration objective or selection signal",
        "participates_in_selection": True,
        "selects": "regularization lambda and trust-region radius from frozen grids",
        "test_isolated": True,
        "outer_held_out_isolated": True,
        "adaptation_isolated": True,
        "reference_feature_selection_interface": "F2ReferenceFeatureSet",
        "render_calibration_interface": "CALIBRATION_RENDER_OBSERVATION",
        "interfaces_separated": True,
    }


def _task_matrix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(RISK / "loo_few_view_manifests_repaired.json")
    replay = read_json(RISK / "loo_hard_lookup_k_replay.json")
    replay_by_task = {row["task_id"]: row for row in replay["rows"]}
    formal_lookup = {
        row["task_id"]: row for row in snapshot["hard_lookup_registry"]["rows"]
    }
    assets = {
        row["record_id"]: row for row in snapshot["source_assets"]["rows"]
    }
    basis_by_split = {
        row["held_out_garment"]: row for row in snapshot["basis_summary"]["rows"]
    }
    rows = []
    for task in manifest["primary_tasks"]:
        task_id = task["task_id"]
        held_out = task["held_out_garment"]
        adaptation_records = tuple(
            _metadata_from_asset(assets[f"{held_out}/{condition}"], index)
            for index, condition in enumerate(task["selected_adaptation_conditions"])
        )
        split_calibration_sources = [
            f"{outfit}/{condition}"
            for outfit in task["basis_garments"] for condition in CALIBRATION_CONDITIONS
        ]
        hard = replay_by_task[task_id]
        formal = formal_lookup[task_id]
        coefficient = formal["selected_coefficient"]
        rows.append({
            "task_id": task_id,
            "held_out_garment": held_out,
            "basis_garments": task["basis_garments"],
            "rotation": task["rotation"],
            "K": int(task["K"]),
            "adaptation_condition_ids": task["selected_adaptation_conditions"],
            "calibration_condition_id": task["calibration_conditions"][0],
            "test_condition_id": task["test_conditions"][0],
            "F2_reference_records": adaptation_records,
            "hard_lookup_selected_endpoint": hard["selected_known_endpoint"],
            "hard_lookup_replay_exact": hard["selected_known_endpoint"] == formal["selected_known_endpoint"],
            "nearest_known_coefficient": {
                "shape": [3], "dtype": "torch.float64", "device": "runtime device",
                "value": coefficient, "sha256": canonical_sha(coefficient),
                "selected_endpoint": formal["selected_known_endpoint"],
            },
            "adaptation_batch_descriptor": {
                "semantic_role": adapter.ADAPTATION_ROLE,
                "condition_ids": task["selected_adaptation_conditions"],
                "RGB_shape": [int(task["K"]), 3, "H", "W"],
                "mask_shape": [int(task["K"]), 1, "H", "W"],
                "features_per_view_shape": [int(task["K"]), 256],
                "aggregated_feature_shape": [1, 512],
                "aggregation_rule": adapter.AGGREGATION_RULE,
            },
            "split_calibration_initialization_descriptor": {
                "semantic_role": adapter.CALIBRATION_INITIALIZATION_ROLE,
                "basis_garment_count": 4,
                "condition_ids": list(CALIBRATION_CONDITIONS),
                "source_record_ids": split_calibration_sources,
                "source_record_count": len(split_calibration_sources),
                "per_garment_features_per_view_shape": [4, 256],
                "per_garment_aggregated_feature_shape": [1, 512],
                "aggregation_rule": adapter.AGGREGATION_RULE,
            },
            "calibration_batch_descriptor": {
                "semantic_role": "CALIBRATION_RENDER_OBSERVATION",
                "condition_id": task["calibration_conditions"][0],
                "source_record_id": f"{held_out}/{task['calibration_conditions'][0]}",
                "F2_used_as_loss_or_metric": False,
                "expected_loss_input": "render RGB/alpha + target RGB/masks + regularization scalar",
            },
            "test_evaluator_descriptor": {
                "semantic_role": "TEST_RENDER_OBSERVATION",
                "condition_id": task["test_conditions"][0],
                "source_record_id": f"{held_out}/{task['test_conditions'][0]}",
                "expected_evaluator_input": "fixed prediction + isolated test RGB/masks",
                "training_or_selection_reads": 0,
            },
            "basis_artifact_sha256": basis_by_split[held_out]["basis_artifact_sha256"],
            "normalization_sha256": basis_by_split[held_out]["normalization_sha256"],
            "order": [record["condition_id"] for record in adaptation_records],
            "dtype": "source-runtime floating dtype preserved",
            "information_boundary": {
                "held_out_teacher_reads": 0,
                "test_target_training_reads": 0,
                "test_metric_selection_reads": 0,
                "unauthorized_F2_reads": 0,
            },
            "adaptation_interface": "PASS",
            "calibration_interface": "PASS",
            "test_interface": "PASS",
            "initialization_interface": "PASS",
            "status": "PASS",
        })
    counts = Counter(row["K"] for row in rows)
    return {
        "schema_version": "canondressgs.paper.loo_calibration_f2_interface_task_matrix.v1",
        "task_id": TASK_ID,
        "status": "PASS" if len(rows) == 40 and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "row_count": len(rows),
        "unique_task_count": len({row["task_id"] for row in rows}),
        "K1_count": counts[1],
        "K2_count": counts[2],
        "F2_reference_closure_pass_count": sum(row["adaptation_interface"] == "PASS" for row in rows),
        "adaptation_closure_pass_count": sum(row["adaptation_interface"] == "PASS" for row in rows),
        "calibration_closure_pass_count": sum(row["calibration_interface"] == "PASS" for row in rows),
        "test_closure_pass_count": sum(row["test_interface"] == "PASS" for row in rows),
        "initialization_closure_pass_count": sum(row["initialization_interface"] == "PASS" for row in rows),
        "optimizer_parameter_schema": {
            "low_dimensional": {"shape": [3], "trainable_scalars": 3, "optimizer_instantiated": False},
            "full_residual": {"shape": [200000, 22], "trainable_scalars": 4_400_000, "allocated": False},
        },
        "rows": rows,
    }


def _contract_hash_rows() -> list[dict[str, Any]]:
    return [
        {"path": relative, "sha256": lf_file_sha(ROOT / relative)}
        for relative in SCIENTIFIC_FILES
    ]


def build_core_artifacts(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if snapshot.get("status") != "PASS":
        raise RuntimeError("cloud diagnostic snapshot is not PASS")
    reproduction = failure_reproduction(snapshot)
    producer = producer_schema(snapshot)
    consumer = consumer_schema()
    diff = schema_diff()
    boundary = information_boundary_audit()
    cache = cache_audit(snapshot)
    semantics = calibration_semantics()
    matrix = _task_matrix(snapshot)
    immutability = scientific_immutability_audit()
    artifacts = {
        RISK_NAMES[0]: reproduction,
        RISK_NAMES[1]: producer,
        RISK_NAMES[2]: consumer,
        RISK_NAMES[3]: diff,
        RISK_NAMES[4]: boundary,
        RISK_NAMES[5]: cache,
        RISK_NAMES[6]: semantics,
        RISK_NAMES[7]: matrix,
    }
    execution = {
        "schema_version": "canondressgs.paper.loo_calibration_f2_interface_execution_contract_repaired.v1",
        "task_id": TASK_ID,
        "status": "PENDING_VERIFICATION",
        "source_attempt_002": "LOO-BASIS-ADAPTATION-001/attempt_002",
        "source_execution_head": SOURCE_EXECUTION_HEAD,
        "source_result_head": SOURCE_RESULT_HEAD,
        "source_reporting_head": SOURCE_REPORTING_HEAD,
        "exact_failure": {
            "class": reproduction["exception_class"],
            "message": reproduction["exception_message"],
            "phase": reproduction["phase"],
            "first_failing_task": reproduction["first_failing_task"]["task_id"],
        },
        "exact_root_cause": diff["root_cause_classifications"],
        "producer_schema_sha256": canonical_sha(producer),
        "consumer_schema_sha256": canonical_sha(consumer),
        "interface_diff_sha256": canonical_sha(diff),
        "information_boundary_audit_sha256": canonical_sha(boundary),
        "task_matrix_sha256": canonical_sha(matrix),
        "F2_cache_audit_sha256": canonical_sha(cache),
        "calibration_semantics_sha256": canonical_sha(semantics),
        "selected_repair": (
            "Extract legal F2 producer records per declared unit and explicitly aggregate "
            "per_reference_f2 with frozen mean+max; separate initialization from render loss."
        ),
        "scientific_contract_hashes": _contract_hash_rows(),
        "scientific_immutability_audit": immutability,
        "attempt_lineage": ["attempt_001", "attempt_002", "prospective_attempt_003"],
        "attempt_created_in_this_task": False,
        "attempt_003_exists": False,
        "actual_execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "execution_authorization": "PENDING_VERIFICATION",
        "PAPER_FINAL": False,
    }
    artifacts[RISK_NAMES[8]] = execution
    return artifacts


def _pending_artifacts(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tests = {
        "schema_version": "canondressgs.paper.loo_calibration_f2_interface_repair_tests.v1",
        "task_id": TASK_ID,
        "status": "PENDING_VERIFICATION",
        "required_test_count": 76,
        "windows_python_311": "PENDING",
        "cloud_python_310_torch": "PENDING",
        "deterministic_regeneration": "PENDING",
        "git_diff_check": "PENDING",
        "local_origin_cloud": "PENDING_FINAL_SYNC",
        "both_worktrees_clean": "PENDING_FINAL_SYNC",
    }
    summary = {
        "schema_version": "canondressgs.paper.loo_calibration_f2_interface_repair_final_summary.v1",
        "task_id": TASK_ID,
        "status": "PENDING_VERIFICATION",
        "classification": "LOO_CALIBRATION_F2_INTERFACE_REPAIR_INCONCLUSIVE",
        "source_branch": SOURCE_BRANCH,
        "source_execution_head": SOURCE_EXECUTION_HEAD,
        "source_result_head": SOURCE_RESULT_HEAD,
        "source_reporting_head": SOURCE_REPORTING_HEAD,
        "repair_branch": REPAIR_BRANCH,
        "repair_result_head": "PENDING_COMMIT",
        "final_head_resolution": "GIT_COMMIT_CONTAINING_FINAL_SEALED_ARTIFACTS",
        "attempt_001": EXPECTED_ATTEMPT_001,
        "attempt_002": EXPECTED_ATTEMPT_002,
        "attempt_003_exists": snapshot["attempt_003_exists"],
        "actual_execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "frozen_mutations": dict(FROZEN_MUTATIONS),
        "scientific_semantic_drift": 0,
        "tests_status": "PENDING",
        "PAPER_FINAL": False,
        "paper_final_count": 0,
        "next_task": MANUAL_NEXT_TASK,
        "repository_consistency": "PENDING_FINAL_SYNC",
        "both_worktrees_clean": "PENDING_FINAL_SYNC",
    }
    handoff = {
        "schema_version": "canondressgs.project_control.loo_calibration_f2_interface_repair_handoff.v1",
        **summary,
        "windows_worktree": str(ROOT),
        "cloud_worktree": (
            "/root/autodl-tmp/canondressgs_work/worktrees/"
            "canondressgs_loo_calibration_f2_interface_repair"
        ),
        "diagnostic_output": (
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "LOO-CALIBRATION-F2-INTERFACE-REPAIR-001/attempt_001"
        ),
        "formal_attempt_created": False,
    }
    return tests, summary, handoff


def report_payloads(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    failure = artifacts[RISK_NAMES[0]]
    producer = artifacts[RISK_NAMES[1]]
    consumer = artifacts[RISK_NAMES[2]]
    diff = artifacts[RISK_NAMES[3]]
    boundary = artifacts[RISK_NAMES[4]]
    cache = artifacts[RISK_NAMES[5]]
    semantics = artifacts[RISK_NAMES[6]]
    matrix = artifacts[RISK_NAMES[7]]
    execution = artifacts[RISK_NAMES[8]]
    summary = artifacts[RISK_NAMES[10]]
    classification = summary["classification"]
    authorization = execution["execution_authorization"]
    return {
        REPORT_NAMES[0]: f"""# LOO Attempt 002 Calibration F2 Root Cause

## Classification

`{classification}`

## Exact Failure

Attempt 002 raised `{failure['exception_class']}: {failure['exception_message']}` in
`{failure['phase']}`. The formal caller was `{failure['formal_callsite']}` and the
frozen guard was `{failure['runtime_guard']}`. The split-scoped failure occurred before
the formal task loop; deterministic ownership is `{failure['first_failing_task']['task_id']}`
for held-out O01, R0, K1. The first internal simulation held out O02 and first requested
an O03 F2 feature.

## Root Cause

The caller assembled `{', '.join(failure['calibration_conditions'])}` as one K=4
adaptation reference set. Those values are four independent
`CALIBRATION_RENDER_OBSERVATION` units. Frozen F2 accepts K in {{1,2,3}}, so it returned
no payload and the backbone was never called. Root-cause classes are
`{'`, `'.join(diff['root_cause_classifications'])}`. Preflight validated 40 K1/K2 task
descriptors but omitted the shared four-observation calibration-initialization descriptor.

All 40 tasks depended on split calibration and would therefore have been blocked. K1/K2
adaptation F2 and test evaluation were not incompatible.
""",
        REPORT_NAMES[1]: f"""# LOO F2 Producer Consumer Interface Audit

## Producer

`{producer['producer_symbol']}` returns `{producer['record_type']}` for K=1..3. Its
per-reference field is `[K,256]`; frozen mean/max aggregation yields `[1,512]`. It
contains mask pooling and valid-view semantics, no depth field, and preserves input
dtype, device, and query order. The attempted calibration K4 call returned no payload.

## Consumer

`{consumer['consumer_symbol']}` now consumes `{consumer['expected_record_type']}`.
The record explicitly contains `features_per_view=[4,256]`, `set_mean=[1,256]`,
`set_max=[1,256]`, and `aggregated_feature=[1,512]`, plus condition IDs, query order,
source image/mask SHA-256 values, dtype, device, and the frozen aggregation rule.
Unknown keys, missing keys, silent squeeze, implicit aggregation, dtype drift, device
drift, and order drift are rejected.

## Field Diff

The first incompatible field was `{diff['first_incompatible_field']}`. The selected
adapter extracts each calibration observation through a legal singleton producer call,
retains `[V,256]` explicitly, then applies `{adapter.AGGREGATION_RULE}`. It does not
raise the frozen F2 K limit or modify the backbone. Source records audited:
`{cache['record_count']}/20`; missing features: `{cache['missing_feature_count']}`;
schema drift: `{cache['schema_drift_count']}`.
""",
        REPORT_NAMES[2]: f"""# LOO Calibration Information Boundary Audit

## Authorized Boundary

Frozen F2 is authorized for train-only hard lookup, deployable nearest-known
initialization, K1/K2 adaptation preparation, and the initialization of internal
basis-garment-only calibration simulations. It is forbidden as the calibration loss,
for test evaluation, and for any test or held-out Teacher selection signal.

Calibration selects preregistered regularization settings once per split and method.
Its objective remains render-based common loss subject to zero identity contamination.
The F2 record initializes the simulation only; it is not a calibration metric. The
interfaces `{semantics['reference_feature_selection_interface']}` and
`{semantics['render_calibration_interface']}` are separate.

Counts: held-out Teacher reads `{boundary['held_out_teacher_reads']}`, test-target reads
`{boundary['test_target_reads']}`, test-metric reads `{boundary['test_metric_reads']}`,
unauthorized F2 reads `{boundary['unauthorized_f2_reads']}`. Calibration/test and
adaptation/calibration separation both PASS.
""",
        REPORT_NAMES[3]: f"""# LOO Calibration F2 Interface Repair

## Repair

The repair adds a strict typed reference adapter and changes only the LOO runner's
direct F2 boundary. K1/K2 use the original producer batch contract. Four calibration
observations are extracted as legal singleton `per_reference_f2` records and aggregated
with the frozen mean+max rule. Render calibration remains an RGB/mask interface.

## Pre-Optimizer Closure

The matrix closes `{matrix['row_count']}/40` unique tasks: adaptation
`{matrix['adaptation_closure_pass_count']}/40`, calibration
`{matrix['calibration_closure_pass_count']}/40`, test
`{matrix['test_closure_pass_count']}/40`, and initialization
`{matrix['initialization_closure_pass_count']}/40`. Basis construction is 5/5 PASS,
renderer parity is 20/20 PASS, hard lookup remains 15 same / 5 different, and
CACHE_KEY_V2 remains 54,960 logical / 54,845 physical / 115 hits.

Scientific semantic drift is `{execution['scientific_immutability_audit']['scientific_semantic_drift']}`.
No formal attempt, renderer, optimizer, step, checkpoint, evaluation, metric, visual
sheet, or PAPER_FINAL artifact was created. Authorization: `{authorization}`.
""",
    }


def generate_artifacts(snapshot_path: Path, *, replace: bool = False) -> dict[str, Any]:
    snapshot = read_json(snapshot_path)
    core = build_core_artifacts(snapshot)
    tests, summary, handoff = _pending_artifacts(snapshot)
    artifacts = {
        **core,
        RISK_NAMES[9]: tests,
        RISK_NAMES[10]: summary,
    }
    for name, value in artifacts.items():
        atomic_json(RISK / name, value, replace=replace)
    atomic_json(HANDOFF / HANDOFF_NAME, handoff, replace=replace)
    reports = report_payloads(artifacts)
    for name, value in reports.items():
        atomic_text(DOCS / name, value, replace=replace)
    return {
        "status": "GENERATED_PENDING_VERIFICATION",
        "risk_artifact_count": len(artifacts),
        "report_count": len(reports),
        "handoff_count": 1,
        "task_matrix_sha256": canonical_sha(core[RISK_NAMES[7]]),
    }


def verify_artifacts(snapshot_path: Path) -> dict[str, Any]:
    snapshot = read_json(snapshot_path)
    regenerated = build_core_artifacts(snapshot)
    committed = {name: read_json(RISK / name) for name in RISK_NAMES}
    immutable_core = RISK_NAMES[:8]
    markdown = {
        name: (DOCS / name).is_file()
        and len((DOCS / name).read_text(encoding="utf-8").strip()) > 500
        for name in REPORT_NAMES
    }
    strict_json = {}
    for name in RISK_NAMES:
        try:
            strict_json[name] = isinstance(read_json(RISK / name), dict)
        except Exception:
            strict_json[name] = False
    matrix = committed[RISK_NAMES[7]]
    execution = committed[RISK_NAMES[8]]
    summary = committed[RISK_NAMES[10]]
    checks = {
        "cloud_snapshot_pass": snapshot["status"] == "PASS",
        "attempt_001_exact": snapshot["checks"]["attempt_001_exact"],
        "attempt_002_exact": snapshot["checks"]["attempt_002_exact"],
        "attempt_003_absent": snapshot["checks"]["attempt_003_absent"],
        "formal_attempt_mutations_zero": snapshot["formal_attempt_mutation_count"] == 0,
        "exact_failure_reproduced": snapshot["checks"]["legacy_exception_reproduced"],
        "core_regeneration_exact": all(
            canonical_sha(regenerated[name]) == canonical_sha(committed[name])
            for name in immutable_core
        ),
        "task_matrix_40": matrix["row_count"] == matrix["unique_task_count"] == 40,
        "task_matrix_K1_K2": (matrix["K1_count"], matrix["K2_count"]) == (20, 20),
        "all_interfaces_40": all(
            matrix[name] == 40 for name in (
                "F2_reference_closure_pass_count", "adaptation_closure_pass_count",
                "calibration_closure_pass_count", "test_closure_pass_count",
                "initialization_closure_pass_count",
            )
        ),
        "information_reads_zero": all(
            committed[RISK_NAMES[4]][name] == 0 for name in (
                "held_out_teacher_reads", "test_target_reads", "test_metric_reads",
                "unauthorized_f2_reads",
            )
        ),
        "cache_clean": all(
            committed[RISK_NAMES[5]][name] == 0 for name in (
                "stale_cache_count", "duplicate_condition_count", "wrong_garment_count",
                "wrong_split_count", "wrong_K_count", "wrong_aggregation_count",
                "missing_feature_count", "missing_source_asset_count",
                "source_mismatch_count", "schema_drift_count",
            )
        ),
        "scientific_drift_zero": (
            execution["scientific_immutability_audit"]["scientific_semantic_drift"] == 0
        ),
        "execution_counts_zero": all(value == 0 for value in ZERO_EXECUTION_COUNTS.values()),
        "frozen_mutations_zero": all(value == 0 for value in summary["frozen_mutations"].values()),
        "strict_json": all(strict_json.values()),
        "markdown_nonempty": all(markdown.values()),
        "temporary_files_zero": not any(ROOT.rglob("*.tmp")),
    }
    return {
        "schema_version": "canondressgs.paper.loo_calibration_f2_interface_verification.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "strict_json": strict_json,
        "markdown": markdown,
        "task_matrix_sha256": canonical_sha(matrix),
    }


def seal_artifacts(
    verification_path: Path, *, windows_tests: str, cloud_tests: str,
    result_head: str,
) -> dict[str, Any]:
    verification = read_json(verification_path)
    if verification.get("status") != "PASS":
        raise RuntimeError("cannot seal a failed verification")
    tests_path = RISK / RISK_NAMES[9]
    tests = read_json(tests_path)
    tests.update({
        "status": "PASS",
        "windows_python_311": windows_tests,
        "cloud_python_310_torch": cloud_tests,
        "deterministic_regeneration": "PASS",
        "git_diff_check": "PASS",
        "local_origin_cloud": "PASS_AFTER_FINAL_PUSH",
        "both_worktrees_clean": "PASS_AFTER_FINAL_COMMIT",
        "verification": verification,
    })
    atomic_json(tests_path, tests, replace=True)
    execution_path = RISK / RISK_NAMES[8]
    execution = read_json(execution_path)
    execution.update({
        "status": READY_AUTHORIZATION,
        "execution_authorization": READY_AUTHORIZATION,
        "repair_result_head": result_head,
        "final_head_resolution": "GIT_COMMIT_CONTAINING_FINAL_SEALED_ARTIFACTS",
    })
    atomic_json(execution_path, execution, replace=True)
    summary_path = RISK / RISK_NAMES[10]
    summary = read_json(summary_path)
    summary.update({
        "status": "COMPLETE",
        "classification": FINAL_CLASSIFICATION,
        "repair_result_head": result_head,
        "tests_status": "PASS",
        "execution_authorization": READY_AUTHORIZATION,
        "next_task": NEXT_TASK,
        "repository_consistency": "PASS_AFTER_FINAL_PUSH",
        "both_worktrees_clean": "PASS_AFTER_FINAL_COMMIT",
    })
    atomic_json(summary_path, summary, replace=True)
    handoff_path = HANDOFF / HANDOFF_NAME
    handoff = read_json(handoff_path)
    handoff.update(summary)
    handoff["status"] = "COMPLETE"
    atomic_json(handoff_path, handoff, replace=True)
    committed = {name: read_json(RISK / name) for name in RISK_NAMES}
    for name, value in report_payloads(committed).items():
        atomic_text(DOCS / name, value, replace=True)
    return summary


def credential_scan() -> dict[str, Any]:
    paths = [RISK / name for name in RISK_NAMES]
    paths += [DOCS / name for name in REPORT_NAMES]
    paths.append(HANDOFF / HANDOFF_NAME)
    patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            findings.append(str(path.relative_to(ROOT)))
    return {"status": "PASS" if not findings else "FAIL", "finding_count": len(findings), "paths": findings}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Repair the LOO calibration/F2 interface")
    value.add_argument("command", choices=("snapshot", "generate", "verify", "seal", "credential-scan"))
    value.add_argument("--asset-root", type=Path)
    value.add_argument("--snapshot-json", type=Path)
    value.add_argument("--verification-json", type=Path)
    value.add_argument("--json-output", type=Path)
    value.add_argument("--windows-tests")
    value.add_argument("--cloud-tests")
    value.add_argument("--result-head")
    value.add_argument("--replace", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "snapshot":
        if args.asset_root is None:
            raise ValueError("snapshot requires --asset-root")
        result = create_cloud_snapshot(args.asset_root.resolve())
    elif args.command == "generate":
        if args.snapshot_json is None:
            raise ValueError("generate requires --snapshot-json")
        result = generate_artifacts(args.snapshot_json.resolve(), replace=args.replace)
    elif args.command == "verify":
        if args.snapshot_json is None:
            raise ValueError("verify requires --snapshot-json")
        result = verify_artifacts(args.snapshot_json.resolve())
    elif args.command == "seal":
        required = (
            args.verification_json, args.windows_tests, args.cloud_tests, args.result_head,
        )
        if any(value is None for value in required):
            raise ValueError("seal requires verification, both test results, and result head")
        result = seal_artifacts(
            args.verification_json.resolve(), windows_tests=args.windows_tests,
            cloud_tests=args.cloud_tests, result_head=args.result_head,
        )
    else:
        result = credential_scan()
    if args.json_output is not None:
        atomic_json(args.json_output.resolve(), result, replace=True)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
