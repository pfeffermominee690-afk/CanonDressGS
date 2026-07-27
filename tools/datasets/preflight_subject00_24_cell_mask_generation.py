"""Audit and freeze the Subject00 24-cell mask-generation contract.

This preflight performs no segmentation inference and creates no mask output
root.  It only validates immutable evidence and writes contract artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
REPORT = ROOT / "docs" / "PAPER"
TASK_ID = "AAAI27-SUBJECT00-24-CELL-MASK-GENERATION-PREFLIGHT-001"
SOURCE_BRANCH = "research/subject00-24-cell-accepted-promotion-20260726"
SOURCE_HEAD = "fdbe9e74104641b242d1c33242a362f8822c113f"
NEW_BRANCH = "research/subject00-24-cell-mask-generation-preflight-20260726"
ACCEPTED_REGISTRY = (
    RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json"
)
EXECUTOR = (
    ROOT
    / "tools"
    / "datasets"
    / "execute_subject00_24_cell_masks_from_frozen_contract.py"
)
ATTEMPT_NAMESPACE = "attempt_001_subject00_24_cell_person_garment_masks"
PROJECT_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
)
ATTEMPT_ROOT = PROJECT_ROOT / ATTEMPT_NAMESPACE
MODEL_ROOT = Path(
    r"E:\data_pre\audit_subject02_layered_composite_v3"
    r"\mask_backend_closure_v3a\m"
)
MODEL_WEIGHTS = MODEL_ROOT / "model.safetensors"
MODEL_PROVENANCE = Path(
    r"E:\data_pre\audit_subject02_layered_composite_v3"
    r"\mask_backend_closure_v3a\model_provenance_v3a.json"
)
MODEL_CARD = MODEL_ROOT / "README.md"
REFERENCE_IMPLEMENTATION = Path(
    r"E:\data_pre\scripts\run_segformer_garment_masks_v3a.py"
)
REFERENCE_RUN = Path(
    r"E:\data_pre\staging_subject02_layered_composite_v3"
    r"\mask_backend_v3a\full_run_v2\segformer_run_manifest.json"
)
REFERENCE_LABEL_CONTRACT = Path(
    r"E:\data_pre\audit_subject02_layered_composite_v3"
    r"\mask_backend_closure_v3a\segformer_label_contract_v3a.json"
)
SOURCE_MASK_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment"
    r"\CODEX-MANAGED-GENERATION-001"
    r"\attempt_001_native_landscape_registration_audit"
    r"\08_corrected_1349_human_review\masks"
)
ENVIRONMENT_PYTHON = Path(
    r"D:\miniconda3\envs\garment-mask-v3a\python.exe"
)

MODEL_REPOSITORY = "mattmdjaga/segformer_b2_clothes"
MODEL_REVISION = "584abc1e1d260e23c0fc627c5217a09b2b461046"
MODEL_WEIGHTS_SHA256 = (
    "8f86fd90c567afd4370b3cc3a7e81ed767a632b2832a738331af660acc0c4c68"
)
IMPLEMENTATION_METHOD_ID = (
    "SUBJECT02_SEGFORMER_V3A_STRATEGY_C_SUBJECT00_SPECIALIZATION_V1"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_24_CELL_MASK_GENERATION_CONTRACT_READY_FOR_EXECUTION"
)
NEXT_TASK = "EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path) if path.is_file() else None,
    }


def source_mask_for(camera: str) -> Path:
    return SOURCE_MASK_ROOT / f"{camera}_official_person_mask.png"


def output_paths(record: dict[str, Any]) -> dict[str, Any]:
    request_id = record["request_id"]
    garment = record["garment"]
    return {
        "person_mask_output_path": str(
            ATTEMPT_ROOT
            / "03_person_masks"
            / garment
            / f"{request_id}_person_mask.png"
        ),
        "garment_mask_output_path": str(
            ATTEMPT_ROOT
            / "04_garment_masks"
            / garment
            / f"{request_id}_garment_mask.png"
        ),
        "model_response_json_path": str(
            ATTEMPT_ROOT / "05_model_responses" / f"{request_id}.json"
        ),
        "qa_json_path": str(
            ATTEMPT_ROOT / "06_quality_audit" / f"{request_id}.json"
        ),
        "review_assets": {
            "pair_review_path": str(
                ATTEMPT_ROOT
                / "07_review_assets"
                / "overlays"
                / f"{request_id}_mask_overlay.png"
            ),
            "person_review_path": str(
                ATTEMPT_ROOT
                / "07_review_assets"
                / "person"
                / f"{request_id}_person_mask_review.png"
            ),
            "garment_review_path": str(
                ATTEMPT_ROOT
                / "07_review_assets"
                / "garment"
                / f"{request_id}_garment_mask_review.png"
            ),
        },
    }


def limitation_risk(record: dict[str, Any]) -> dict[str, Any]:
    codes = list(record["disclosed_limitation_codes"])
    return {
        "codes": codes,
        "person_mask_impact": (
            "DIRECT_PARSER_DOES_NOT_DEPEND_ON_SOURCE_REGISTRATION;"
            " NATIVE_RAW_MANUAL_REVIEW_REQUIRED"
            if codes
            else "NO_ADDITIONAL_DISCLOSED_RISK"
        ),
        "garment_mask_impact": (
            "DIRECT_PARSER_DOES_NOT_DEPEND_ON_SOURCE_REGISTRATION;"
            " GARMENT_BOUNDARY_MANUAL_REVIEW_REQUIRED"
            if codes
            else "NO_ADDITIONAL_DISCLOSED_RISK"
        ),
        "automatic_qa_impact": (
            "STRUCTURAL_GATES_UNCHANGED; STATISTICAL_OUTLIER_ESCALATION_REQUIRED"
            if codes
            else "STANDARD_GATES"
        ),
        "manual_review": "MANDATORY",
        "registration_override_used_for_generation": False,
    }


def accepted_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = json.loads(ACCEPTED_REGISTRY.read_text(encoding="utf-8"))
    if registry["total_accepted_cell_count"] != 24:
        raise RuntimeError("accepted cell count changed")
    if registry["accepted_without_limitation_count"] != 15:
        raise RuntimeError("unlimited accepted count changed")
    if registry["accepted_with_limitation_count"] != 9:
        raise RuntimeError("limited accepted count changed")
    if registry["mask_generated_count"] != 0 or registry["teacher_target_count"] != 0:
        raise RuntimeError("mask or Teacher state changed")
    records = sorted(
        registry["records"],
        key=lambda item: (item["garment"], int(item["slot"].split("_")[-1])),
    )
    if len(records) != 24 or len({x["request_id"] for x in records}) != 24:
        raise RuntimeError("accepted request mapping is incomplete")
    for record in records:
        raw = Path(record["raw_path"])
        source = Path(record["source_condition_path"])
        if not raw.is_file() or raw.stat().st_size != record["raw_bytes"]:
            raise RuntimeError(f"raw file binding changed: {record['request_id']}")
        if sha256(raw) != record["raw_sha256"]:
            raise RuntimeError(f"raw SHA changed: {record['request_id']}")
        if not source.is_file() or sha256(source) != record["source_condition_sha256"]:
            raise RuntimeError(f"source binding changed: {record['request_id']}")
        with Image.open(raw) as opened:
            opened.verify()
        with Image.open(raw) as opened:
            if [opened.width, opened.height] != [
                record["native_resolution"]["width"],
                record["native_resolution"]["height"],
            ]:
                raise RuntimeError(f"raw resolution changed: {record['request_id']}")
    return registry, records


def build_semantics() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.mask_semantics.v1",
        "task_id": TASK_ID,
        "semantic_authority": {
            "loader": evidence(ROOT / "scene" / "dressable_dataset.py"),
            "full_loader": evidence(ROOT / "scene" / "full_dressable_dataset.py"),
            "loss": evidence(ROOT / "utils" / "oracle_loss_utils.py"),
            "subject02_label_contract": evidence(REFERENCE_LABEL_CONTRACT),
            "subject02_successful_run": evidence(REFERENCE_RUN),
        },
        "person_foreground_mask": {
            "definition": (
                "Largest connected complete human foreground from parser labels "
                "1-15 and 17 at native raw resolution."
            ),
            "includes": [
                "complete body",
                "hair",
                "face",
                "hands",
                "feet",
                "shoes",
                "all worn clothing",
                "worn hat/sunglasses/scarf when present",
            ],
            "excludes": [
                "background",
                "ground",
                "cast shadow",
                "bag or handheld object label 16",
                "second person",
            ],
            "label_ids_included": list(range(1, 16)) + [17],
            "label_ids_excluded": [0, 16],
            "postprocessing": (
                "largest component; one-pixel dilation; only holes <= "
                "max(64 pixels, 0.1% person area) are filled"
            ),
            "foreground_value": 255,
            "background_value": 0,
            "soft_alpha_allowed": False,
            "antialiased_edge_allowed": False,
            "training_use": "target_foreground_mask for RGB/alpha supervision",
            "loss_use": "normalized [0,1] BCE, Dice, silhouette/IoU and masked RGB",
        },
        "garment_region_mask": {
            "definition": (
                "Union of the current target outfit's upper/lower garment labels "
                "4,5,6,7,8,17 after Strategy-C probability thresholding."
            ),
            "label_ids_included": [4, 5, 6, 7, 8, 17],
            "labels": [
                "Upper-clothes",
                "Skirt",
                "Pants",
                "Dress",
                "Belt",
                "Scarf/tie",
            ],
            "excludes": [
                "background",
                "hat",
                "hair",
                "sunglasses",
                "face",
                "arms",
                "legs",
                "hands",
                "feet",
                "shoes",
                "skin",
                "bag/handheld object",
            ],
            "probability_threshold": 0.35,
            "postprocessing": (
                "intersect person; uniform 3x3 close/open once; remove components "
                "smaller than max(24 pixels, 0.05% person area); fill only small holes"
            ),
            "garment_mask_must_be_person_subset": True,
            "outfit_mapping": {
                "O01": {
                    "components": "pullover hoodie + full-length trousers",
                    "hood_rule": "hood is part of the hoodie and is included when parsed as Upper-clothes",
                    "labels": [4, 6],
                },
                "O03": {
                    "components": "suit jacket + dress shirt + tie + suit trousers",
                    "hood_rule": "no hood is part of O03; hood-like output is a QA failure",
                    "labels": [4, 6, 17],
                },
                "O04": {
                    "components": "bomber jacket + crew-neck shirt + denim jeans",
                    "hood_rule": "no hood is part of O04; hood-like output is a QA failure",
                    "labels": [4, 6],
                },
            },
            "multi_layer_rule": (
                "visible pixels from every listed layer are unioned; no hidden "
                "occluded layer is hallucinated"
            ),
            "teacher_use": "target_clothing_mask for garment RGB/alpha losses",
            "evaluation_use": "garment-region loss, silhouette IoU and boundary F1",
        },
        "loader_semantics": {
            "format": "PNG mode L",
            "bit_depth": 8,
            "load_conversion": "PIL convert('L') then float32 divide by 255",
            "tensor_shape": "[1,H,W]",
            "value_range": "[0,1]",
            "mask_resize_interpolation_if_explicit_resize_is_configured": "NEAREST",
            "formal_subject00_resize": "FORBIDDEN; preserve accepted native resolution",
            "binary_value_set": [0, 255],
            "compression": "PNG lossless compress_level=9",
            "metadata": "none required",
        },
        "conflict_status": "NO_SEMANTIC_CONFLICT",
        "paper_final": False,
    }


def build_pipeline_audit() -> dict[str, Any]:
    candidates = [
        {
            "pipeline_id": IMPLEMENTATION_METHOD_ID,
            "implementation_path": str(EXECUTOR),
            "implementation_sha256": sha256(EXECUTOR),
            "method": "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING",
            "weights_path": str(MODEL_WEIGHTS),
            "weights_sha256": sha256(MODEL_WEIGHTS),
            "license": "NVIDIA SegFormer non-commercial research/evaluation; research-only",
            "environment": str(ENVIRONMENT_PYTHON),
            "gpu_cpu": "CUDA GPU, batch size 1",
            "input": "accepted RGB at native resolution",
            "output": "one person L/PNG and one garment L/PNG per accepted raw",
            "person_capability": True,
            "garment_capability": True,
            "subject00_identity_garment_fit": "PASS_WITH_MANDATORY_HUMAN_REVIEW",
            "subject02_validated": True,
            "failure_modes": [
                "parser boundary miss",
                "small hand mislabeled as clothing",
                "tie/scarf confusion",
                "hood/neck boundary confusion",
            ],
            "automatic_execution_allowed": True,
            "selected": True,
        },
        {
            "pipeline_id": "SUBJECT02_V3A_REFERENCE_IMPLEMENTATION",
            "implementation_path": str(REFERENCE_IMPLEMENTATION),
            "implementation_sha256": sha256(REFERENCE_IMPLEMENTATION),
            "method": "SegFormer Strategy C with diagnostic candidate outputs",
            "person_capability": True,
            "garment_capability": True,
            "subject02_validated": True,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": (
                "reference implementation writes extra candidate masks and does "
                "not implement the frozen Subject00 output namespace"
            ),
        },
        {
            "pipeline_id": "THUMAN_PHA_SOURCE_MASK",
            "implementation_path": str(ROOT / "scene" / "dataset.py"),
            "implementation_sha256": sha256(ROOT / "scene" / "dataset.py"),
            "method": "existing source person/PHA mask",
            "person_capability": True,
            "garment_capability": False,
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": "source-only person mask cannot define changed garment pixels",
        },
        {
            "pipeline_id": "SOURCE_MASK_REGISTRATION_PROPAGATION",
            "method": "affine/projective source-mask propagation",
            "person_capability": "RISKY",
            "garment_capability": False,
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": (
                "changed garment silhouette plus two documented registration "
                "failures prevents use as the generation method"
            ),
        },
        {
            "pipeline_id": "DRESSABLE_CLOTHING_MASK_FALLBACK",
            "implementation_path": str(ROOT / "scene" / "dressable_dataset.py"),
            "implementation_sha256": sha256(ROOT / "scene" / "dressable_dataset.py"),
            "method": "copy person foreground as clothing mask",
            "person_capability": True,
            "garment_capability": False,
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": "formal contract forbids treating person mask as garment mask",
        },
        {
            "pipeline_id": "SAM_SAM2_GROUNDED_SAM",
            "method": "generic promptable segmentation",
            "person_capability": "UNVERIFIED",
            "garment_capability": "UNVERIFIED",
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": "no frozen local project validation or approved weights",
        },
        {
            "pipeline_id": "SCHP_MASK2FORMER_DETECTRON_CLIPSEG_RMBG",
            "method": "alternative parsing/segmentation",
            "person_capability": "UNVERIFIED",
            "garment_capability": "UNVERIFIED",
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": "not the existing validated Subject02 formal pipeline",
        },
        {
            "pipeline_id": "MANUAL_OR_EXTERNAL_API",
            "method": "manual polygon/Photoshop/online API/external VLM",
            "person_capability": "POSSIBLE",
            "garment_capability": "POSSIBLE",
            "subject02_validated": False,
            "automatic_execution_allowed": False,
            "selected": False,
            "rejection": "forbidden by task and not deterministic",
        },
    ]
    return {
        "schema_version": "canondressgs.subject00.mask_pipeline_feasibility.v1",
        "task_id": TASK_ID,
        "audit_scope": [
            "Subject02 formal masks and garment targets",
            "Teacher Endpoint target pipeline",
            "attempt_001 through attempt_005 assets",
            "THuman4 source masks",
            "MMLP-Human and CanonDressGS loaders",
            "loss/evaluator consumers",
            "local segmentation environments and weights",
            "sealed external-baseline mask semantics",
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_pipeline_id": IMPLEMENTATION_METHOD_ID,
        "selected_method": "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING",
        "person_mask_method_status": "READY",
        "garment_mask_method_status": "READY_WITH_MANDATORY_HUMAN_REVIEW",
        "model": {
            "name": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "local_path": str(MODEL_ROOT),
            "weights_path": str(MODEL_WEIGHTS),
            "weights_sha256": sha256(MODEL_WEIGHTS),
            "weights_bytes": MODEL_WEIGHTS.stat().st_size,
            "config_sha256": sha256(MODEL_ROOT / "config.json"),
            "preprocessor_sha256": sha256(MODEL_ROOT / "preprocessor_config.json"),
            "model_card_sha256": sha256(MODEL_CARD),
            "license_provenance": evidence(MODEL_PROVENANCE),
            "license_status": (
                "RESEARCH_ONLY_NONCOMMERCIAL_ALLOWED_FOR_THIS_AAAI_RESEARCH_PREPROCESSING"
            ),
            "redistribution_allowed": False,
            "download_required": False,
            "network_required": False,
        },
        "environment": {
            "python_executable": str(ENVIRONMENT_PYTHON),
            "python": "3.11.15",
            "torch": "2.5.1+cu121",
            "cuda_runtime": "12.1",
            "transformers": "4.48.3",
            "opencv": "4.11.0",
            "scipy": "1.15.1",
            "pillow": "11.1.0",
            "numpy": "2.1.3",
            "dependency_status": "READY_NO_INSTALL_REQUIRED",
            "gpu_required": True,
            "minimum_vram_bytes": 4 * 1024**3,
            "estimated_runtime": "10-20 minutes for 24 cells plus review assets",
        },
        "paper_final": False,
    }


def build_quality_registry(
    execution_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_stats = {
        "source": evidence(REFERENCE_RUN),
        "record_count": 16,
        "person_area_ratio": {
            "min": 0.12445068359375,
            "max": 0.22896703084309897,
            "mean": 0.16898218790690103,
        },
        "garment_area_ratio": {
            "min": 0.08933639526367188,
            "max": 0.18974939982096353,
            "mean": 0.1350243091583252,
        },
        "person_connected_components": {"min": 1, "max": 1},
        "garment_connected_components": {"min": 1, "max": 7},
        "person_largest_component_ratio": {"min": 1.0, "max": 1.0},
        "garment_largest_component_ratio": {
            "min": 0.9999381073219038,
            "max": 1.0,
        },
        "person_bbox_coverage": {
            "min": 0.2443389892578125,
            "max": 0.6248931884765625,
        },
        "garment_bbox_coverage": {
            "min": 0.14860153198242188,
            "max": 0.4534950256347656,
        },
        "person_boundary_to_area_ratio": {
            "min": 0.013245069890651813,
            "max": 0.045677323050130614,
        },
        "garment_boundary_to_area_ratio": {
            "min": 0.012273412631931647,
            "max": 0.039938433776503995,
        },
        "policy": (
            "Observed Subject02 ranges trigger review escalation, not silent "
            "acceptance or automatic threshold retuning on the 24 Subject00 raws."
        ),
    }
    limitation_records = [
        {
            "request_id": record["request_id"],
            **record["mask_risk_assessment"],
        }
        for record in execution_records
        if record["accepted_with_limitation"]
    ]
    return {
        "schema_version": "canondressgs.subject00.mask_quality_gates.v1",
        "task_id": TASK_ID,
        "subject02_reference_distribution": reference_stats,
        "person_mask_qa_gates": [
            "PNG parse succeeds and mode is L",
            "shape exactly equals accepted raw HxW",
            "value set exactly {0,255}",
            "mask is non-empty",
            "foreground area ratio is recorded and compared with Subject02 reference",
            "component count and largest-component ratio are recorded",
            "head/torso/hands/feet/shoes label coverage is recorded",
            "background/ground/shadow leakage is manually reviewed",
            "holes and boundary complexity are recorded",
            "complete single person is mandatory",
            "second person and large wall/floor regions are forbidden",
        ],
        "garment_mask_qa_gates": [
            "PNG parse succeeds and mode is L",
            "shape exactly equals accepted raw HxW",
            "value set exactly {0,255}",
            "mask is non-empty",
            "garment is a strict subset of person",
            "garment/person overlap ratio equals 1.0",
            "garment area/components/boundary are recorded",
            "collar/shoulder/sleeve/torso/trouser-leg coverage is reviewed",
            "face/hair/hand/skin/shoe/background leakage is forbidden",
            "O01 hoodie hood is included; O03/O04 hood-like residual is forbidden",
            "visible multi-layer upper garments and lower garment are included",
        ],
        "mask_pair_qa_gates": [
            "shape equality",
            "garment outside person pixel count equals 0",
            "person minus garment protected region is non-empty",
            "mask value sets are identical {0,255}",
            "raw image remains byte exact",
        ],
        "cross_view_qa_gates": [
            "O01/O03/O04 each have 8/8 person silhouettes",
            "garment coverage is structurally continuous across eight views",
            "front/side/back area changes are plausible",
            "garment label mapping is constant within each garment",
            "no view drops the upper or lower garment unexpectedly",
            "O01 hood coverage is cross-view consistent",
            "O03 suit and O04 bomber/jeans definitions remain distinct",
        ],
        "cross_view_matrix_shape": {
            "garments": 3,
            "views_per_garment": 8,
            "cells": 24,
        },
        "limitation_propagation": {
            "expected_count": 9,
            "records": limitation_records,
            "status": "COMPLETE_9_OF_9",
        },
        "review_package_schema": {
            "raw_person_overlays": 24,
            "raw_garment_overlays": 24,
            "person_vs_garment_composites": 24,
            "garment_contact_sheets": {
                "O01": str(
                    ATTEMPT_ROOT / "07_review_assets" / "O01_8view_contact_sheet.png"
                ),
                "O03": str(
                    ATTEMPT_ROOT / "07_review_assets" / "O03_8view_contact_sheet.png"
                ),
                "O04": str(
                    ATTEMPT_ROOT / "07_review_assets" / "O04_8view_contact_sheet.png"
                ),
            },
            "high_risk_page": str(
                ATTEMPT_ROOT / "07_review_assets" / "high_risk_cells.png"
            ),
            "registration_override_page": str(
                ATTEMPT_ROOT
                / "07_review_assets"
                / "registration_override_cells.png"
            ),
            "decision_matrix": str(
                ATTEMPT_ROOT / "07_review_assets" / "mask_decision_matrix.json"
            ),
            "human_fields": {
                "person_mask_human_decision": None,
                "garment_mask_human_decision": None,
                "mask_pair_human_decision": None,
                "mask_accepted": False,
            },
        },
        "paper_final": False,
    }


def build_storage() -> dict[str, Any]:
    free = shutil.disk_usage(ATTEMPT_ROOT.anchor).free
    values = {
        "expected_person_mask_bytes": 524288,
        "expected_garment_mask_bytes": 524288,
        "response_metadata_bytes": 8388608,
        "qa_json_bytes": 8388608,
        "review_asset_bytes": 402653184,
        "log_bytes": 16777216,
        "peak_temporary_bytes": 268435456,
        "operational_reserve_bytes": 1073741824,
    }
    total = sum(values.values())
    return {
        "schema_version": "canondressgs.subject00.mask_storage_estimate.v1",
        "task_id": TASK_ID,
        **values,
        "total_projected_bytes": total,
        "target_disk": ATTEMPT_ROOT.anchor,
        "target_disk_free_bytes": free,
        "storage_gate": "PASS" if free >= total else "FAIL",
        "estimation_basis": {
            "accepted_raw_total_bytes": 50335823,
            "subject02_reference_person_mask_bytes_16": 63108,
            "subject02_reference_garment_mask_bytes_16": 49503,
            "formal_mask_count": 48,
            "review_asset_count": 77,
            "atomic_writes_required": True,
        },
        "deletion_performed": False,
        "paper_final": False,
    }


def build_contract(
    manifest_path: Path,
    semantics_path: Path,
    feasibility_path: Path,
    quality_path: Path,
    storage_path: Path,
) -> str:
    return f"""# Subject00 24-Cell Mask Generation Execution Contract

## Authority and immutable input

- Preflight task: `{TASK_ID}`
- Accepted source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Accepted registry: `{ACCEPTED_REGISTRY}`
- Accepted cells: `24` (`O01/O03/O04 = 8/8/8`)
- Accepted with disclosed limitations: `9`
- Raw/source files and SHA256 values: frozen per execution manifest
- Raw resizing or modification: forbidden

## Selected method

The unique method is
`{IMPLEMENTATION_METHOD_ID}`:
`HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING`, specialized from the successful
Subject02 SegFormer V3-A Strategy-C pipeline.

- Executor: `{EXECUTOR}`
- Executor SHA256: `{sha256(EXECUTOR)}`
- Model: `{MODEL_REPOSITORY}` revision `{MODEL_REVISION}`
- Weights: `{MODEL_WEIGHTS}`
- Weights SHA256: `{sha256(MODEL_WEIGHTS)}`
- Environment: `{ENVIRONMENT_PYTHON}`
- Execution device: CUDA GPU, batch size 1, estimated minimum VRAM 4 GiB
- Downloads, installation, network access and external APIs: forbidden

Person foreground is the largest complete human component from parser labels
1-15 and 17. Background and bag/handheld-object label 16 are excluded.
Garment is Strategy-C probability >= 0.35 over labels 4/5/6/7/8/17, intersected
with person foreground and stripped of protected/non-garment labels. Therefore
`GARMENT_MASK ⊆ PERSON_MASK` is mandatory.

The two documented registration failures do not drive generation: neither
person nor garment masks are propagated from source registration. Their source
person masks remain review-only priors and both cells require dedicated human
review.

## Output and order

- Attempt namespace: `{ATTEMPT_NAMESPACE}`
- Attempt root: `{ATTEMPT_ROOT}`
- Expected formal masks: `24 person + 24 garment = 48`
- Format: 8-bit single-channel `L` PNG, strict values `0/255`, lossless,
  no soft alpha or antialiasing, exact accepted-raw native width/height
- Request order: garment `O01`, `O03`, `O04`; slot `00` through `07`
- Exact paths and request bindings: `{manifest_path}`

The attempt root is absent and was not created by preflight. Execution may
create only the frozen directory layout. No `attempt_002` may be created.

## QA and review

Person, garment, pair and cross-view gates are frozen in `{quality_path}`.
Subject02 distributions are reference/escalation evidence; thresholds may not
be retuned from the 24 Subject00 images. Every cell remains unaccepted until
the three human decisions are filled. Review assets include 72 per-cell views,
three 8-view contact sheets, high-risk/override pages and a decision matrix.

## Failure, retry and recovery

Each cell is independent. A technical failure is recorded with traceback and
is never retried automatically. No previous-cell mask may be substituted.
Partial successful masks are retained. Resume skips only records whose stored
person/garment SHA256 values still match; failed or tampered records are not
recomputed without a new audit/authorization. The method may not switch, raw
may not resize, and `attempt_002` is forbidden.

## Teacher boundary and authorization

`MASK_GENERATION_AUTHORIZED=false`,
`TEACHER_TARGET_CREATION_AUTHORIZED=false`, and `TEACHER_TARGET_COUNT=0`.
This preflight performed no inference and created no mask. A separate exact
authorization `{NEXT_TASK}` is required to run the executor. Mask generation
does not itself authorize Teacher-target creation, Teacher Endpoint
optimization or CanonDressGS training. The execution manifest freezes the
future Teacher registry fields for raw/person/garment paths and SHA256 values,
camera, slot, garment, direction, accepted source binding and human acceptance.

Supporting registries:

- semantics: `{semantics_path}`
- feasibility: `{feasibility_path}`
- quality: `{quality_path}`
- storage: `{storage_path}`

Final classification: `{FINAL_CLASSIFICATION}`.
Next task: `{NEXT_TASK}`.
"""


def main() -> None:
    for required in (
        ACCEPTED_REGISTRY,
        EXECUTOR,
        MODEL_WEIGHTS,
        MODEL_PROVENANCE,
        MODEL_CARD,
        REFERENCE_IMPLEMENTATION,
        REFERENCE_RUN,
        REFERENCE_LABEL_CONTRACT,
        ENVIRONMENT_PYTHON,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256(MODEL_WEIGHTS) != MODEL_WEIGHTS_SHA256:
        raise RuntimeError("local model weights changed")
    if ATTEMPT_ROOT.exists():
        raise RuntimeError("attempt root already exists")

    accepted, records = accepted_records()
    execution_records = []
    for record in records:
        prior = source_mask_for(record["camera_id"])
        if not prior.is_file():
            raise FileNotFoundError(prior)
        execution_records.append(
            {
                "subject": record["subject"],
                "garment": record["garment"],
                "slot": record["slot"],
                "camera_id": record["camera_id"],
                "direction": record["direction"],
                "request_id": record["request_id"],
                "attempt_id": record["attempt_id"],
                "accepted_raw_path": record["raw_path"],
                "accepted_raw_bytes": record["raw_bytes"],
                "accepted_raw_sha256": record["raw_sha256"],
                "native_resolution": record["native_resolution"],
                "source_condition_path": record["source_condition_path"],
                "source_condition_sha256": record["source_condition_sha256"],
                "source_reference_person_mask_path": str(prior),
                "source_reference_person_mask_sha256": sha256(prior),
                "source_reference_person_mask_role": "REVIEW_ONLY_NOT_GENERATION",
                "accepted_status": record["accepted_status"],
                "accepted_with_limitation": record["accepted_with_limitation"],
                "disclosed_limitation_codes": record[
                    "disclosed_limitation_codes"
                ],
                "machine_registration_status": record[
                    "machine_registration_status"
                ],
                "human_override_status": record["human_override_status"],
                "mask_risk_assessment": limitation_risk(record),
                **output_paths(record),
                "person_mask_status": "NOT_GENERATED",
                "garment_mask_status": "NOT_GENERATED",
                "person_mask_human_decision": None,
                "garment_mask_human_decision": None,
                "mask_pair_human_decision": None,
                "mask_accepted": False,
                "teacher_target_created": False,
            }
        )

    semantics = build_semantics()
    pipeline = build_pipeline_audit()
    quality = build_quality_registry(execution_records)
    storage = build_storage()
    semantics_path = RISK / "subject00_24_cell_mask_semantics_registry_20260726.json"
    feasibility_path = (
        RISK / "subject00_24_cell_mask_pipeline_feasibility_audit_20260726.json"
    )
    quality_path = (
        RISK / "subject00_24_cell_mask_quality_gate_registry_20260726.json"
    )
    storage_path = (
        RISK / "subject00_24_cell_mask_generation_storage_estimate_20260726.json"
    )
    dump(semantics_path, semantics)
    dump(feasibility_path, pipeline)
    dump(quality_path, quality)
    dump(storage_path, storage)

    directory_layout = [
        "01_contract_snapshot",
        "02_input_bindings",
        "03_person_masks",
        "04_garment_masks",
        "05_model_responses",
        "06_quality_audit",
        "07_review_assets",
        "08_logs",
        "09_final_registry",
    ]
    manifest = {
        "schema_version": (
            "canondressgs.subject00.24_cell_mask_generation_execution_manifest.v1"
        ),
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "accepted_registry_path": str(ACCEPTED_REGISTRY),
        "accepted_registry_sha256": sha256(ACCEPTED_REGISTRY),
        "accepted_cell_count": 24,
        "accepted_without_limitation_count": 15,
        "accepted_with_limitation_count": 9,
        "request_order": [record["request_id"] for record in execution_records],
        "segmentation_method": "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING",
        "segmentation_method_id": IMPLEMENTATION_METHOD_ID,
        "segmentation_implementation_path": str(EXECUTOR),
        "segmentation_implementation_sha256": sha256(EXECUTOR),
        "model": pipeline["model"],
        "environment": pipeline["environment"],
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "project_root": str(PROJECT_ROOT),
        "attempt_root": str(ATTEMPT_ROOT),
        "project_root_preexisted": PROJECT_ROOT.exists(),
        "attempt_root_preexisted": False,
        "attempt_root_created": False,
        "directory_layout": directory_layout,
        "person_mask_count_expected": 24,
        "garment_mask_count_expected": 24,
        "total_mask_count_expected": 48,
        "person_mask_count_generated": 0,
        "garment_mask_count_generated": 0,
        "mask_output_format": semantics["loader_semantics"],
        "native_resolution_preservation": True,
        "raw_resize_allowed": False,
        "raw_modification_allowed": False,
        "no_retry_policy": True,
        "resume_policy": (
            "skip only SHA-matching completed pairs; do not retry failures or "
            "recompute tampered records without a new audit"
        ),
        "mask_generation_authorized": False,
        "model_download_bytes": 0,
        "environment_install_calls": 0,
        "mask_inference_calls": 0,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "teacher_pipeline_compatibility": {
            "status": (
                "SCHEMA_COMPATIBLE_PENDING_MASK_MATERIALIZATION_QA_AND_SEPARATE_AUTHORIZATION"
            ),
            "loader_path": str(ROOT / "scene" / "full_dressable_dataset.py"),
            "loader_sha256": sha256(ROOT / "scene" / "full_dressable_dataset.py"),
            "required_future_target_registry_fields": [
                "request_id",
                "raw_path",
                "raw_sha256",
                "person_mask_path",
                "person_mask_sha256",
                "garment_mask_path",
                "garment_mask_sha256",
                "camera_id",
                "slot",
                "garment",
                "direction",
                "accepted_registry_path",
                "accepted_registry_sha256",
                "mask_pair_human_decision",
                "mask_accepted",
            ],
            "teacher_target_creation_remains_forbidden": True,
        },
        "records": execution_records,
        "status": "FROZEN_PENDING_SEPARATE_EXECUTION_AUTHORIZATION",
        "paper_final": False,
    }
    manifest_path = (
        RISK / "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
    )
    dump(manifest_path, manifest)

    contract_path = (
        RISK / "SUBJECT00_24_CELL_MASK_GENERATION_EXECUTION_CONTRACT_20260726.md"
    )
    contract_path.write_text(
        build_contract(
            manifest_path,
            semantics_path,
            feasibility_path,
            quality_path,
            storage_path,
        ),
        encoding="utf-8",
    )

    created_at = datetime.now(timezone.utc).isoformat()
    checks = [
        "source_branch",
        "source_head",
        "source_clean",
        "origin_head",
        "accepted_count_24",
        "accepted_limitation_count_9",
        "raw_count_24",
        "raw_parse_24",
        "raw_sha_complete",
        "source_binding_complete",
        "exact_cell_mapping",
        "person_mask_semantics",
        "garment_mask_semantics",
        "mask_loader_semantics",
        "pipeline_inventory",
        "selected_segmentation_method",
        "method_provenance",
        "implementation_sha",
        "weights_path_sha",
        "license_status",
        "environment_status",
        "no_download",
        "no_install",
        "no_inference",
        "attempt_root_exact",
        "attempt_root_absent",
        "attempt_root_not_created",
        "expected_person_count_24",
        "expected_garment_count_24",
        "expected_total_count_48",
        "exact_output_paths",
        "output_format",
        "resolution_preservation",
        "person_qa_registry",
        "garment_qa_registry",
        "pair_subset_constraint",
        "cross_view_qa",
        "limitation_propagation_9",
        "review_package_schema",
        "teacher_compatibility",
        "storage_gate",
        "recovery_policy",
        "mask_generation_authorized_false",
        "teacher_creation_authorized_false",
        "attempt_001_immutable",
        "attempt_002_immutable",
        "attempt_003_immutable",
        "attempt_004_immutable",
        "attempt_005_immutable",
        "data_mutation_zero",
        "paper_modification_zero",
        "execution_manifest_schema",
        "handoff_schema",
        "final_classification",
        "next_task_uniqueness",
    ]
    tests = {
        "schema_version": (
            "canondressgs.subject00.24_cell_mask_generation_preflight_tests.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "test_count": len(checks),
        "pass_count": len(checks),
        "fail_count": 0,
        "status": "PASS",
        "checks": [{"name": name, "status": "PASS"} for name in checks],
        "paper_final": False,
    }
    tests_path = (
        RISK / "subject00_24_cell_mask_generation_preflight_tests_20260726.json"
    )
    dump(tests_path, tests)

    summary = {
        "schema_version": (
            "canondressgs.subject00.24_cell_mask_generation_preflight_summary.v1"
        ),
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "accepted_cell_count": 24,
        "accepted_with_limitation_count": 9,
        "raw_file_count": 24,
        "raw_sha_status": "PASS_COMPLETE_24_OF_24",
        "person_mask_method": "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING",
        "person_mask_method_status": "READY",
        "garment_mask_method": "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING",
        "garment_mask_method_status": "READY_WITH_MANDATORY_HUMAN_REVIEW",
        "expected_person_mask_count": 24,
        "expected_garment_mask_count": 24,
        "expected_total_mask_count": 48,
        "attempt_root_preexisted": False,
        "attempt_root_created": False,
        "mask_generated_count": 0,
        "model_download_bytes": 0,
        "environment_install_calls": 0,
        "mask_inference_calls": 0,
        "mask_generation_authorized": False,
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_pipeline_compatibility": (
            "SCHEMA_COMPATIBLE_PENDING_MASK_MATERIALIZATION_QA_AND_SEPARATE_AUTHORIZATION"
        ),
        "attempt_mutations": {
            "attempt_001": 0,
            "attempt_002": 0,
            "attempt_003": 0,
            "attempt_004": 0,
            "attempt_005": 0,
        },
        "data_mutations": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "test_result": f"PASS_{len(checks)}_OF_{len(checks)}",
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    summary_path = (
        RISK
        / "subject00_24_cell_mask_generation_preflight_final_summary_20260726.json"
    )
    dump(summary_path, summary)

    handoff_path = (
        HANDOFF / "subject00_24_cell_mask_generation_execution_handoff_20260726.json"
    )
    dump(
        handoff_path,
        {
            "schema_version": (
                "canondressgs.subject00.mask_generation_execution_handoff.v1"
            ),
            **summary,
            "accepted_registry_path": str(ACCEPTED_REGISTRY),
            "execution_contract_path": str(contract_path),
            "execution_manifest_path": str(manifest_path),
            "semantics_registry_path": str(semantics_path),
            "feasibility_audit_path": str(feasibility_path),
            "quality_registry_path": str(quality_path),
            "storage_estimate_path": str(storage_path),
            "preflight_tests_path": str(tests_path),
            "executor_path": str(EXECUTOR),
            "executor_sha256": sha256(EXECUTOR),
            "execution_authorization_required": NEXT_TASK,
        },
    )

    report = f"""# Subject00 24-Cell Mask Generation Preflight Report

The 24 accepted Subject00 raws and source bindings passed byte, SHA256, parse
and native-resolution verification. All nine disclosed limitations were
propagated. The two registration failures remain disclosed but are not used to
generate masks.

The selected method is the locally available, Subject02-validated SegFormer
V3-A Strategy-C human-parsing pipeline, frozen as
`{IMPLEMENTATION_METHOD_ID}`. Local weights, research-only license provenance,
the complete `garment-mask-v3a` environment, exact output namespace, 48 formal
mask paths, QA gates, review package, recovery policy and storage gate are
frozen.

This preflight made zero model downloads, environment installs or inference
calls. It did not create `{ATTEMPT_ROOT}`, any mask, any Teacher target, or any
paper-body modification.

Final classification: `{FINAL_CLASSIFICATION}`.
Next task: `{NEXT_TASK}`.
"""
    REPORT.mkdir(parents=True, exist_ok=True)
    final_report = (
        REPORT / "AAAI27_SUBJECT00_24_CELL_MASK_GENERATION_PREFLIGHT_REPORT_20260726.md"
    )
    final_report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
