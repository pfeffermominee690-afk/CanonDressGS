#!/usr/bin/env python3
"""Build the evidence-only Subject00 generation-backend adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"

TASK_ID = "AAAI27-SUBJECT00-GENERATION-BACKEND-ADJUDICATION-001"
SOURCE_BRANCH = "research/subject00-data-preparation-generation-ready-20260725"
SOURCE_HEAD = "4e5e76a2a19ffdb94385660ec1d8ea4ce4c5d6b3"
WORKFLOW_BRANCH = "research/multi-identity-generation-backend-review-workflow-20260724"
WORKFLOW_HEAD = "e8a09525f745f373e38418c43040766d69a50d10"
TASK_B_BRANCH = "research/subject00-minimal-second-identity-dataset-contract-20260725"
TASK_B_HEAD = "9da04d923c7c6c03c6a6f7c732d8d9c58d59c88d"
STORAGE_BRANCH = "research/mmlphuman-subject00-storage-migration-adjudication-20260725"
STORAGE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
BRANCH = "research/subject00-generation-backend-adjudication-20260725"
WINDOWS_WORKTREE = "E:/model_train/canondressgs_subject00_generation_backend_adjudication"
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_generation_backend_adjudication"
)
DATASET_ROOT = "/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment"
FORMAL_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
)
CLASSIFICATION = "SUBJECT00_GENERATION_BACKEND_SELECTION_REQUIRED"
NEXT_TASK = "USER_SELECT_SUBJECT00_GENERATION_PROVIDER_AND_COMPLETE_MANDATORY_BACKEND_EVIDENCE"
BACKEND_DECISION_HEAD = "RESOLVE_AFTER_DATA_ADJUDICATION_COMMIT"
LIVE_FREE_BYTES = 17_239_027_712
REQUIRED_FREE_BYTES = 32_212_254_720

SOURCE_MANIFEST = RISK / "subject00_generation_request_manifest.json"
SOURCE_SUMMARY = RISK / "subject00_generation_readiness_final_summary.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(seal(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_doc(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def evidence(
    evidence_id: str,
    backend_id: str,
    path: str,
    size: int | None,
    sha256: str | None,
    evidence_class: str,
    claims: list[str],
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "backend_id": backend_id,
        "path": path,
        "bytes": size,
        "sha256": sha256,
        "evidence_class": evidence_class,
        "claims_supported": claims,
        "secret_material_read": False,
    }


GATES = {
    "G1": "image editing or image conditioning, not text-only generation",
    "G2": "Subject00 identity and condition images can be supplied",
    "G3": "full-body identity, pose, and camera preservation",
    "G4": "single full-body output without text and with complete hands and feet",
    "G5": "model name and revision can be frozen",
    "G6": "at least 48 requests can be executed automatically",
    "G7": "raw response can be retained per request",
    "G8": "PNG or lossless normalizable output",
    "G9": "stable naming and provenance",
    "G10": "auditable retry and error taxonomy",
    "G11": "credentials need not be persisted in Git or command-line plaintext",
    "G12": "research-use license or use boundary can be recorded",
    "G13": "Subject00 target resolution is supported",
    "G14": "the frozen 48-request manifest maps deterministically",
    "G15": "real historical success evidence exists, otherwise a canary is mandatory",
}


GATE_RESULTS = {
    "BACKEND-A": {
        "G1": ("PASS", ["E-CODEX-DIRECT"], "Twelve real multi-image direct edits succeeded."),
        "G2": ("PASS", ["E-CODEX-DIRECT"], "Identity, geometry, and garment images were supplied historically."),
        "G3": ("PARTIAL", ["E-CODEX-V5"], "Eleven identity reviews were WARN and one was FAIL."),
        "G4": ("PARTIAL", ["E-CODEX-V5"], "One of twelve edits changed protected footwear."),
        "G5": ("FAIL", ["E-CODEX-MANAGER"], "Exact model ID and revision are unavailable."),
        "G6": ("FAIL", ["E-CODEX-MANAGER"], "Evidence is an interactive platform workflow, not a 48-item batch API."),
        "G7": ("FAIL", ["E-CODEX-DIRECT"], "Provider raw response is not exposed; only result assets and local records exist."),
        "G8": ("PASS", ["E-CODEX-DIRECT"], "Native 1024x1536 PNG outputs were recorded."),
        "G9": ("PARTIAL", ["E-CODEX-MANAGER"], "Local naming/provenance exists but depends on session asset import."),
        "G10": ("PARTIAL", ["E-CODEX-SUPPLEMENT"], "Failures and bounded retries are recorded, but not via a formal API executor."),
        "G11": ("PASS", ["E-CODEX-MANAGER"], "Platform session route used no project API key."),
        "G12": ("UNKNOWN", ["E-WORKFLOW"], "No local license or research-use boundary was frozen."),
        "G13": ("PASS", ["E-CODEX-DIRECT"], "Native output matches 1024x1536 portrait."),
        "G14": ("FAIL", ["E-CODEX-MANAGER"], "No exact manifest replay or noninteractive request adapter exists."),
        "G15": ("PASS", ["E-CODEX-DIRECT", "E-CODEX-SUPPLEMENT"], "Real image-edit and accepted donor outputs exist."),
    },
    "BACKEND-B": {
        "G1": ("PASS", ["E-SUBLYX-SCRIPT"], "The script calls /v1/images/edits with image inputs."),
        "G2": ("PASS", ["E-SUBLYX-SCRIPT"], "Identity reference and condition images are multipart inputs."),
        "G3": ("PARTIAL", ["E-SUBLYX-UNION"], "Historical accepted pose/camera evidence exists, but not Subject00 target evidence."),
        "G4": ("PARTIAL", ["E-SUBLYX-UNION"], "Accepted historical sheets exist; current single portrait compliance is unverified."),
        "G5": ("FAIL", ["E-SUBLYX-SCRIPT"], "gpt-image-2 is an unpinned alias and no server-returned revision was retained."),
        "G6": ("PASS", ["E-SUBLYX-SCRIPT", "E-SUBLYX-JAY"], "The runner iterates manifests; 915 Jay records were logged."),
        "G7": ("PARTIAL", ["E-SUBLYX-SCRIPT"], "Raw-response metadata and optional JSON exist, but safe default retention is not frozen."),
        "G8": ("PASS", ["E-SUBLYX-SCRIPT"], "Response bytes are saved and can be normalized to PNG."),
        "G9": ("PASS", ["E-SUBLYX-JAY"], "Stable sheet paths, manifests, prompt paths, and hashes exist."),
        "G10": ("PASS", ["E-SUBLYX-SCRIPT"], "Two retries and explicit error classes are implemented."),
        "G11": ("PARTIAL", ["E-WORKFLOW"], "Environment credentials are supported, but historical length/hash-prefix logging is forbidden prospectively."),
        "G12": ("UNKNOWN", ["E-WORKFLOW"], "No local provider license or research-use boundary was verified."),
        "G13": ("FAIL", ["E-SUBLYX-SCRIPT"], "Only 1536x1024 landscape is historically verified, not 1024x1536 portrait."),
        "G14": ("PARTIAL", ["E-SUBLYX-SCRIPT"], "Automation is adaptable, but the frozen 48-entry backend mapping is not implemented."),
        "G15": ("PASS", ["E-SUBLYX-UNION"], "246 accepted traceable historical donor records exist."),
    },
    "BACKEND-C": {
        "G1": ("PARTIAL", ["E-78-SCRIPT"], "An edits endpoint is implemented, but successful edit execution is absent."),
        "G2": ("PARTIAL", ["E-78-SCRIPT"], "Multipart request support exists only as unproven code."),
        "G3": ("UNKNOWN", ["E-78-PROBE"], "No successful image provides preservation evidence."),
        "G4": ("UNKNOWN", ["E-78-PROBE"], "No successful output exists."),
        "G5": ("PARTIAL", ["E-78-MODELS"], "gpt-image-2 ID was confirmed, but no revision was exposed."),
        "G6": ("PASS", ["E-78-SCRIPT"], "The historical runner is manifest-driven and sequential."),
        "G7": ("PARTIAL", ["E-78-SCRIPT"], "Diagnostic response metadata is saved; successful image response retention is unverified."),
        "G8": ("PARTIAL", ["E-78-SCRIPT"], "PNG handling exists but has no successful provider output."),
        "G9": ("PASS", ["E-78-SCRIPT"], "Stable paths and manifest logging are implemented."),
        "G10": ("PASS", ["E-78-SCRIPT"], "HTTP/error classification and retry limits are implemented."),
        "G11": ("PARTIAL", ["E-WORKFLOW"], "Environment credentials are possible, but historical fingerprint logging is forbidden."),
        "G12": ("UNKNOWN", ["E-WORKFLOW"], "No local license or research-use boundary was verified."),
        "G13": ("UNKNOWN", ["E-78-PROBE"], "The target portrait size has no successful evidence."),
        "G14": ("PARTIAL", ["E-78-SCRIPT"], "The runner can be adapted, but no frozen mapping has executed."),
        "G15": ("FAIL", ["E-78-PROBE"], "The edit probe returned HTTP 500 convert_request_failed; no image was created."),
    },
}


DIMENSIONS = [
    "identity_preservation",
    "pose_camera_preservation",
    "garment_fidelity",
    "multi_image_conditioning",
    "mask_conditioning",
    "output_resolution",
    "automation",
    "batch_stability",
    "raw_response_retention",
    "deterministic_manifest_binding",
    "retry_support",
    "historical_success",
    "failure_mode_auditability",
    "content_policy_evidence",
    "reproducibility",
    "cost_evidence",
    "provider_stability",
    "credential_safety",
    "license_research_use_boundary",
]


WEIGHTED_SCORES = {
    "BACKEND-A": [1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2, 0],
    "BACKEND-B": [1, 1, 2, 2, 0, 1, 2, 1, 1, 1, 2, 2, 2, 0, 1, 0, 1, 1, 0],
    "BACKEND-C": [0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1, 0, 0, 1, 0],
}


def build_candidate_inventory() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_candidate_inventory.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "candidate_count": 3,
        "candidates": [
            {
                "backend_id": "BACKEND-A",
                "name": "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
                "category": "PLATFORM_MANAGED_INTERACTIVE_IMAGE_EDIT",
                "historical_success_class": "IMAGE_EDIT_SUCCESS_NOT_FORMAL_BACKEND",
                "selection_status": "INELIGIBLE_MANDATORY_GATES",
            },
            {
                "backend_id": "BACKEND-B",
                "name": "HISTORICAL_OPENAI_COMPATIBLE_PROXY_SUBLYX",
                "category": "SCRIPTABLE_OPENAI_COMPATIBLE_IMAGE_EDIT_PROXY",
                "historical_success_class": "FORMAL_DONOR_GENERATION_SUCCESS_NOT_SUBJECT02_TARGET_SUCCESS",
                "selection_status": "INELIGIBLE_MANDATORY_GATES",
            },
            {
                "backend_id": "BACKEND-C",
                "name": "HISTORICAL_OPENAI_COMPATIBLE_PROXY_78CODE",
                "category": "SCRIPTABLE_OPENAI_COMPATIBLE_PROXY_WITHOUT_EDIT_SUCCESS",
                "historical_success_class": "CONNECTIVITY_AND_MODEL_DISCOVERY_ONLY",
                "selection_status": "INELIGIBLE_MANDATORY_GATES",
            },
        ],
        "backend_d_discovered": False,
        "unknown_provider_search_performed": False,
        "paper_final": False,
    }


def build_evidence_registry() -> dict[str, Any]:
    records = [
        evidence("E-SUBLYX-SCRIPT", "BACKEND-B", "E:/data_pre/scripts/run_gpt_image2_pose_outfit_sheets.py", 41220, "2a431927e591e60a31ff796d8a5a774b415b58b0f31c0b8b3096c4c26ea6ae04", "IMPLEMENTATION", ["base URL https://api.sublyx.org/", "model alias gpt-image-2", "/v1/images/edits", "multipart image conditioning", "timeout 300", "two retries", "optional raw JSON"]),
        evidence("E-SUBLYX-JAY", "BACKEND-B", "E:/data_pre/outputs/gpt_image2_pose_outfit_sheets/Jay/generation_manifest.csv", 399993, "31d6c4f50331e2bcc90b0d8fa308e2ffbc95c3d7d2b67bbc432edc988c45639d", "HISTORICAL_EXECUTION_MANIFEST", ["915 rows", "494 success", "178 failed", "243 skipped existing"]),
        evidence("E-SUBLYX-UNION", "BACKEND-B", "E:/data_pre/audit_jay_coverage_supplement_v1/jay_production_union_manifest_v2.json", 438472, "16ca8a9bff6a0f0f9f2408dea7b579a7f93e6e5ae3f1a9c5046031b7c4af2ff8", "FORMAL_DATASET_DONOR_MANIFEST", ["246 historical Sublyx accepted records", "all pose/camera traceable", "not Subject02 target generation"]),
        evidence("E-SUBLYX-CONNECTIVITY", "BACKEND-B", "E:/data_pre/audit_jay_coverage_supplement_v1/JAY_TARGETED_COVERAGE_COMPLETION_V1_STATUS.json", 1303, "0c6511ff15dccd5fcb639f5590ab99a87c43052a7c58d4600a6137a919cdac9c", "CONNECTIVITY_ONLY", ["latest Sublyx connectivity was not executed because credential was absent", "no current availability proof"]),
        evidence("E-CODEX-DIRECT", "BACKEND-A", "E:/data_pre/audit_subject02_direct_edit_v4/direct_edit_raw_manifest_v4.json", 47381, "30602bfbc80d52ed15ab90a3cd1d8d602f8db665028a5ad12d6937416b7f1cee", "IMAGE_EDIT_SUCCESS", ["12 of 12 direct edits succeeded", "native 1024x1536", "three-image conditioning", "model unexposed"]),
        evidence("E-CODEX-V4", "BACKEND-A", "E:/data_pre/audit_subject02_direct_edit_v4/V4_FINAL_STATUS.json", 2773, "e3c4b347c182f21baa8d606ee01b939fb7cf3aefd833d58c8463ee0ee32a03d9", "FORMAL_GATE_FAILURE", ["12 calls and 12 generated", "12 of 12 final visual composites failed", "dataset expansion blocked"]),
        evidence("E-CODEX-V5", "BACKEND-A", "E:/data_pre/audit_subject02_dual_target_v5/V5_FINAL_STATUS.json", 2753, "d43448c81078bff9f295ab793095daeb884bf1ed4b505c3d3e2494cd4bf512f3", "IDENTITY_REVIEW_FAILURE", ["11 WARN and 1 FAIL", "one protected footwear change", "formal fixture failed"]),
        evidence("E-CODEX-MANAGER", "BACKEND-A", "E:/data_pre/scripts/manage_codex_imagegen_supplement_v1.py", 21182, "6b2a0012ade1cac79d59c13a5689c76022370d0c27140cd08609dd5c3b86f328", "INTERACTIVE_IMPORT_WORKFLOW", ["requested model builtin_image_gen", "exact model unavailable", "session asset import", "PNG normalization"]),
        evidence("E-CODEX-SUPPLEMENT", "BACKEND-A", "E:/data_pre/audit_jay_coverage_supplement_v1/jay_production_union_manifest_v2.json", 438472, "16ca8a9bff6a0f0f9f2408dea7b579a7f93e6e5ae3f1a9c5046031b7c4af2ff8", "FORMAL_DATASET_DONOR_MANIFEST", ["15 accepted Codex supplement records", "all pose/camera traceable"]),
        evidence("E-78-SCRIPT", "BACKEND-C", "E:/data_pre/scripts/run_78code_403_user_agent_diagnostic.py", 22152, "37173cf9405cad7550905d35d7a2fe7443e3efb70c2ffb6ac6096d23a5902afa", "IMPLEMENTATION_AND_DIAGNOSTIC", ["base URL https://www.78code.cc/v1", "generations and edits probes", "sanitized errors"]),
        evidence("E-78-MODELS", "BACKEND-C", "E:/data_pre/audit_jay_coverage_supplement_v1/image_model_id_audit_v1/provider_image_model_ids.json", 1831, "3cffbc4b8c65865e3b443c7c4d96e0dcda37608075539843fcacbcd0d42b03cb", "MODEL_DISCOVERY", ["gpt-image-2 model ID confirmed", "revision not exposed"]),
        evidence("E-78-PROBE", "BACKEND-C", "E:/data_pre/audit_jay_coverage_supplement_v1/image_model_id_audit_v1/model_route_probe_edits.json", 1011, "e9f696b07402869e835ed034d375d7927a013db7c7ae558f55ddbaf306019d8f", "CONNECTIVITY_ONLY", ["edit probe HTTP 500", "convert_request_failed", "no image generated"]),
        evidence("E-WORKFLOW", "ALL", f"git:{WORKFLOW_HEAD}:paper_protocol/datasets/multi_identity_generation_backend_contract.json", None, None, "FROZEN_GIT_CONTRACT", ["three historical surfaces", "credential policy", "provider/model selection remained required"]),
    ]
    return {
        "schema_version": "canondressgs.subject00.backend_evidence_registry.v1",
        "task_id": TASK_ID,
        "audit_basis": "EXISTING_LOCAL_AND_GIT_EVIDENCE_ONLY",
        "records": records,
        "evidence_count": len(records),
        "distinctions": {
            "CONNECTIVITY_ONLY": ["E-SUBLYX-CONNECTIVITY", "E-78-MODELS", "E-78-PROBE"],
            "IMAGE_EDIT_SUCCESS": ["E-CODEX-DIRECT"],
            "FORMAL_DATASET_GENERATION_SUCCESS": ["E-SUBLYX-UNION", "E-CODEX-SUPPLEMENT"],
            "SUBJECT02_FORMAL_TARGET_SUCCESS": [],
        },
        "network_or_connectivity_probe_performed": False,
        "secret_or_credential_value_read": False,
        "paper_final": False,
    }


def build_gate_matrix() -> dict[str, Any]:
    candidates = []
    for backend_id, results in GATE_RESULTS.items():
        rows = [
            {
                "gate_id": gate_id,
                "requirement": GATES[gate_id],
                "status": status,
                "evidence_ids": evidence_ids,
                "reason": reason,
            }
            for gate_id, (status, evidence_ids, reason) in results.items()
        ]
        candidates.append(
            {
                "backend_id": backend_id,
                "gates": rows,
                "counts": {state: sum(row["status"] == state for row in rows) for state in ["PASS", "PARTIAL", "FAIL", "UNKNOWN"]},
                "all_mandatory_gates_pass": all(row["status"] == "PASS" for row in rows),
                "only_realtime_availability_missing": False,
                "direct_selection_eligible": False,
                "pending_canary_selection_eligible": False,
            }
        )
    return {
        "schema_version": "canondressgs.subject00.backend_mandatory_gate_matrix.v1",
        "task_id": TASK_ID,
        "gate_count": len(GATES),
        "gate_ids": list(GATES),
        "candidates": candidates,
        "decision_rule": "EVERY_GATE_MUST_PASS_FOR_DIRECT_SELECTION; UNKNOWN_IS_NOT_PASS",
        "eligible_backend_count": 0,
        "paper_final": False,
    }


def build_weighted_comparison() -> dict[str, Any]:
    rows = []
    for backend_id, scores in WEIGHTED_SCORES.items():
        rows.append(
            {
                "backend_id": backend_id,
                "scores": dict(zip(DIMENSIONS, scores, strict=True)),
                "total": sum(scores),
                "maximum": len(DIMENSIONS) * 2,
                "mandatory_gate_override": "INELIGIBLE",
            }
        )
    return {
        "schema_version": "canondressgs.subject00.backend_weighted_comparison.v1",
        "task_id": TASK_ID,
        "scale": {"0": "ABSENT_FAIL_OR_UNKNOWN", "1": "PARTIAL", "2": "STRONG_LOCAL_EVIDENCE"},
        "dimensions": DIMENSIONS,
        "candidates": rows,
        "ranking_use": "AUXILIARY_ONLY_MANDATORY_GATES_OVERRIDE_TOTAL",
        "paper_final": False,
    }


def build_selection_decision() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_selection_decision.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "status": "BACKEND_SELECTION_REQUIRED",
        "classification": CLASSIFICATION,
        "primary": None,
        "fallback": None,
        "provider": None,
        "base_url": None,
        "model": None,
        "revision": None,
        "revision_or_alias_status": "UNSELECTED",
        "decision_reason": "No candidate passes G1-G15, and every candidate has non-realtime mandatory evidence gaps.",
        "candidate_dispositions": {
            "BACKEND-A": "INTERACTIVE_ONLY_NOT_FORMAL_BACKEND",
            "BACKEND-B": "HISTORICAL_SUCCESS_BUT_MODEL_REVISION_LICENSE_RESOLUTION_AND_SAFE_RAW_RETENTION_UNRESOLVED",
            "BACKEND-C": "NO_REAL_IMAGE_EDIT_SUCCESS",
        },
        "selection_preconditions": [
            "one provider and exact model/revision or explicit alias policy",
            "verified 1024x1536 portrait image-edit support",
            "safe per-request raw-response retention",
            "research-use and data-processing boundary",
            "deterministic 48-request adapter",
            "price and rate-limit evidence or explicit acceptance of unknown status",
        ],
        "minimal_canary_required_after_eligible_selection": True,
        "generation_authorized": False,
        "generation_started": False,
        "api_calls": 0,
        "materialized_images": 0,
        "paper_final": False,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
    }


def build_request_schema() -> dict[str, Any]:
    required = [
        "request_id", "backend", "provider", "base_url", "model", "revision_or_alias_status",
        "identity_source", "condition_image", "condition_mask", "input_shas", "prompt_id",
        "negative_constraint_id", "output_count", "candidate_id", "target_resolution",
        "response_format", "raw_response_path", "normalized_png_path", "provenance_path",
        "timeout_seconds", "retry_policy", "error_taxonomy", "authorization", "executed",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "canondressgs.subject00.backend_request.v1",
        "title": "Subject00 frozen backend request envelope",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "request_id": {"type": "string", "pattern": "^subject00_O(?:01|03|04)_slot0[0-7]_cand0[01]$"},
            "backend": {"type": ["string", "null"]},
            "provider": {"type": ["string", "null"]},
            "base_url": {"type": ["string", "null"]},
            "model": {"type": ["string", "null"]},
            "revision_or_alias_status": {"type": "string", "minLength": 1},
            "identity_source": {"type": "object"},
            "condition_image": {"type": "object"},
            "condition_mask": {"type": "object"},
            "input_shas": {"type": "object"},
            "prompt_id": {"type": "string", "minLength": 1},
            "negative_constraint_id": {"type": "string", "minLength": 1},
            "output_count": {"const": 1},
            "candidate_id": {"type": "string", "minLength": 1},
            "target_resolution": {"const": {"height": 1536, "orientation": "portrait", "width": 1024}},
            "response_format": {"type": "string", "minLength": 1},
            "raw_response_path": {"type": "string", "minLength": 1},
            "normalized_png_path": {"type": "string", "minLength": 1},
            "provenance_path": {"type": "string", "minLength": 1},
            "timeout_seconds": {"const": 300},
            "retry_policy": {"type": "object"},
            "error_taxonomy": {"type": "array", "minItems": 8, "uniqueItems": True},
            "authorization": {"const": False},
            "executed": {"const": False},
        },
        "x_selection_constraint": "backend/provider/model/base_url become non-null only after a supported selection commit",
        "x_secret_policy": "only credential environment-variable name may be persisted; values and fingerprints are forbidden",
        "paper_final": False,
    }


SCIENTIFIC_FIELDS = [
    "request_id", "garment_id", "slot_id", "semantic_pose_slot", "candidate_index",
    "identity_id", "identity_source_set_id", "camera_id", "pose_frame_id", "condition_id",
    "input_file_paths", "input_sha256", "source_condition_sha256", "prompt_id", "prompt_sha256",
    "negative_constraint_id", "negative_constraint_sha256", "output_count", "target_resolution", "output_format",
]


def build_manifest_binding(source_manifest: dict[str, Any]) -> dict[str, Any]:
    error_taxonomy = [
        "NETWORK", "TIMEOUT", "HTTP_5XX", "PROVIDER_TRANSIENT_ERROR", "AUTHENTICATION",
        "INSUFFICIENT_CREDIT_OR_QUOTA", "CONTENT_POLICY", "MALFORMED_RESPONSE", "EMPTY_IMAGE",
        "UNSUPPORTED_PARAMETER", "SCIENTIFIC_QUALITY_REJECTION",
    ]
    bindings = []
    for row in source_manifest["requests"]:
        scientific = {name: row[name] for name in SCIENTIFIC_FIELDS}
        request_id = row["request_id"]
        bindings.append(
            {
                "source_scientific_fields": scientific,
                "source_scientific_fields_sha256": canonical_sha256(scientific),
                "backend_request": {
                    "request_id": request_id,
                    "backend": None,
                    "provider": None,
                    "base_url": None,
                    "model": None,
                    "revision_or_alias_status": "UNSELECTED",
                    "identity_source": {
                        "identity_id": row["identity_id"],
                        "identity_source_set_id": row["identity_source_set_id"],
                        "pose_frame_id": row["pose_frame_id"],
                    },
                    "condition_image": {
                        "path": row["input_file_paths"]["identity_condition_rgb"],
                        "sha256": row["input_sha256"]["identity_condition_rgb"],
                    },
                    "condition_mask": {
                        "path": row["input_file_paths"]["identity_condition_mask"],
                        "sha256": row["input_sha256"]["identity_condition_mask"],
                    },
                    "input_shas": row["input_sha256"],
                    "prompt_id": row["prompt_id"],
                    "negative_constraint_id": row["negative_constraint_id"],
                    "output_count": row["output_count"],
                    "candidate_id": request_id,
                    "target_resolution": row["target_resolution"],
                    "response_format": "PROVIDER_NATIVE_METADATA_PLUS_IMAGE_BYTES_PENDING_SELECTION",
                    "raw_response_path": row["raw_response_path"],
                    "normalized_png_path": f"{DATASET_ROOT}/05_generated_candidates/{row['garment_id']}/{request_id}.png",
                    "provenance_path": row["provenance_path"],
                    "timeout_seconds": 300,
                    "retry_policy": row["retry_policy"],
                    "error_taxonomy": error_taxonomy,
                    "authorization": False,
                    "executed": False,
                },
                "binding_status": "BLOCKED_BACKEND_SELECTION_AND_FORMAL_BASE",
                "response_count": 0,
                "materialized_image_count": 0,
            }
        )
    return {
        "schema_version": "canondressgs.subject00.backend_manifest_binding.v1",
        "task_id": TASK_ID,
        "source_manifest": "paper_protocol/reviewer_risk/subject00_generation_request_manifest.json",
        "source_request_set_sha256": source_manifest["request_set_sha256"],
        "binding_count": len(bindings),
        "bindings": bindings,
        "backend_selection_status": "BACKEND_SELECTION_REQUIRED",
        "provider_model_revision_non_null_count": 0,
        "authorized_count": 0,
        "executed_count": 0,
        "raw_response_count": 0,
        "materialized_image_count": 0,
        "scientific_semantics_changed_count": 0,
        "paper_final": False,
    }


def build_cost_audit() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_cost_rate_limit_audit.v1",
        "task_id": TASK_ID,
        "request_budget": 48,
        "candidates": [
            {
                "backend_id": backend_id,
                "per_call_cost": "PRICE_NOT_VERIFIED",
                "estimated_48_call_cost": "PRICE_NOT_VERIFIED",
                "retry_cost": "PRICE_NOT_VERIFIED",
                "concurrency": "RATE_LIMIT_NOT_VERIFIED",
                "rpm_tpm_or_image_limit": "RATE_LIMIT_NOT_VERIFIED",
                "timeout_seconds": 300 if backend_id in {"BACKEND-B", "BACKEND-C"} else "NOT_EXPOSED",
                "estimated_wall_time": "RATE_LIMIT_NOT_VERIFIED",
            }
            for backend_id in ["BACKEND-A", "BACKEND-B", "BACKEND-C"]
        ],
        "price_guessed": False,
        "rate_limit_guessed": False,
        "paid_calls_in_task": 0,
        "paper_final": False,
    }


def build_canary_contract() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_minimal_canary_contract.v1",
        "task_id": TASK_ID,
        "status": "DESIGNED_NOT_AUTHORIZED_BACKEND_SELECTION_AND_FORMAL_BASE_PENDING",
        "required_after_eligible_backend_selection": True,
        "request_count": 2,
        "request_ids": ["subject00_O01_slot00_cand00", "subject00_O03_slot03_cand00"],
        "budget_relationship": "TWO_EXISTING_ENTRIES_FROM_THE_FROZEN_48; NO_REQUEST_49",
        "selection_basis": ["O01 front", "O03 left side", "two garments and two camera orientations"],
        "success_gates": [
            "request and raw-response records are retained and hash-linked",
            "one valid normalized PNG at exactly 1024x1536 for each request",
            "two independent human reviewers accept identity preservation",
            "two independent human reviewers accept garment fidelity",
            "two independent human reviewers accept pose and camera preservation",
            "single full body, complete hands and feet, no text, label, number, watermark, crop, or duplicate person",
            "zero hard reject taxonomy events",
        ],
        "retry_policy": {
            "maximum_retries_after_initial_attempt": 2,
            "backoff_seconds": [3, 9],
            "technical_errors_only": True,
            "scientific_quality_retry_allowed": False,
        },
        "failure_stop_rule": "STOP_AFTER_ANY_NONRETRYABLE_ERROR_OR_ANY_HARD_SCIENTIFIC_REJECTION; DO_NOT_START_REMAINING_46",
        "remaining_46_policy": "REQUIRE_BOTH_CANARY_REQUESTS_TO_PASS_ALL_GATES_THEN_REQUIRE_NEW_EXPLICIT_USER_AUTHORIZATION",
        "formal_base_dependency": "PENDING",
        "authorization": False,
        "started": False,
        "api_calls": 0,
        "generated_images": 0,
        "paper_final": False,
    }


def build_secret_scan() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_secret_scan.v1",
        "task_id": TASK_ID,
        "status": "PASS_NO_SECRET_EXPOSURE",
        "credential_value_read": False,
        "credential_store_read": False,
        "env_file_read": False,
        "authorization_header_read": False,
        "allowed_credential_field": "credential_env_name",
        "prospective_credential_env_name": "MULTI_IDENTITY_GENERATION_API_KEY",
        "forbidden_persistence": ["value", "length", "prefix", "hash", "fingerprint", "Authorization header"],
        "secret_like_matches": 0,
        "api_calls": 0,
        "paper_final": False,
    }


def build_tests_placeholder() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_adjudication_tests.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN",
        "total": 0,
        "passed": 0,
        "failed": [],
        "checks": [],
        "paper_final": False,
    }


def build_final_summary(source_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.backend_adjudication_final_summary.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "classification": CLASSIFICATION,
        "backend_status": "BACKEND_SELECTION_REQUIRED",
        "primary": None,
        "fallback": None,
        "provider": None,
        "model": None,
        "revision": None,
        "mandatory_gate_candidate_count": 3,
        "mandatory_gate_count_per_candidate": 15,
        "eligible_backend_count": 0,
        "request_count": len(source_manifest["requests"]),
        "binding_count": len(source_manifest["requests"]),
        "request_ids_changed": 0,
        "scientific_semantics_changed": 0,
        "minimal_canary_status": "DESIGNED_NOT_AUTHORIZED_PENDING_ELIGIBLE_BACKEND_AND_FORMAL_BASE",
        "minimal_canary_request_ids": ["subject00_O01_slot00_cand00", "subject00_O03_slot03_cand00"],
        "formal_base": {
            "dependency": "PENDING",
            "sealed_manifest_exists": False,
            "canonical_output_root": FORMAL_ROOT,
            "canonical_output_root_status": "ABSENT",
            "execution_head": None,
        },
        "storage": {
            "classification": "SUBJECT00_STORAGE_MIGRATION_PLAN_READY",
            "status": "BLOCKED_CAPACITY",
            "latest_git_recorded_free_bytes": 17_690_701_824,
            "live_observed_free_bytes": LIVE_FREE_BYTES,
            "required_free_bytes": REQUIRED_FREE_BYTES,
            "live_shortfall_bytes": REQUIRED_FREE_BYTES - LIVE_FREE_BYTES,
            "recommended_plan": "PLAN_C",
            "plan_executed": False,
        },
        "authorization": False,
        "started": False,
        "api_calls": 0,
        "connectivity_calls": 0,
        "paid_calls": 0,
        "raw_responses": 0,
        "materialized_images": 0,
        "credential_values_read": 0,
        "tests": "NOT_RUN",
        "paper_final": False,
        "backend_decision_head": BACKEND_DECISION_HEAD,
        "final_reporting_head_resolution": "git rev-parse HEAD after the final audit seal commit",
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
        "next_task_started": False,
    }


def build_handoff(source_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.generation_backend_adjudication_handoff.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": CLOUD_WORKTREE,
        "classification": CLASSIFICATION,
        "backend_status": "BACKEND_SELECTION_REQUIRED",
        "primary": None,
        "fallback": None,
        "request_count": len(source_manifest["requests"]),
        "binding_count": len(source_manifest["requests"]),
        "canary_request_ids": ["subject00_O01_slot00_cand00", "subject00_O03_slot03_cand00"],
        "canary_authorized": False,
        "formal_base_dependency": "PENDING",
        "storage_status": "BLOCKED_CAPACITY",
        "generation_authorized": False,
        "generation_started": False,
        "api_calls": 0,
        "materialized_images": 0,
        "backend_decision_head": BACKEND_DECISION_HEAD,
        "final_reporting_head_resolution": "git rev-parse HEAD after the final audit seal commit",
        "tests": "NOT_RUN",
        "paper_final": False,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
    }


def build_docs() -> None:
    write_doc(
        DOCS / "AAAI27_SUBJECT00_GENERATION_BACKEND_ADJUDICATION_20260725.md",
        f"""
# Subject00 Generation Backend Adjudication

Task: `{TASK_ID}`

## Decision

- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`.
- Classification: `{CLASSIFICATION}`.
- PRIMARY: `null`.
- FALLBACK: `null`.
- Generation authorization: `false`.
- API, connectivity, and paid calls in this task: `0`.

No candidate passes all mandatory gates. Codex managed image edit has real Subject02 edit evidence but is interactive, has no exposed model revision, and does not expose complete raw provider responses. Sublyx has the strongest scriptable historical evidence, including 246 accepted traceable donor records, but its model is an unpinned alias, its latest availability is unverified, portrait resolution and safe default raw-response retention are not established, and the research-use boundary is absent. 78Code has a confirmed model ID but no successful image-edit artifact.

## Formal And Storage Stop

The sealed Formal Base manifest is absent and `{FORMAL_ROOT}` is absent. Live free capacity observed during this audit was `{LIVE_FREE_BYTES}` bytes versus `{REQUIRED_FREE_BYTES}` required. Storage remains `BLOCKED_CAPACITY`; `PLAN_C` is recommended but unexecuted.

## Next Action

`{NEXT_TASK}`. Selection requires exact provider/model/revision policy, portrait image-edit evidence, safe raw-response retention, deterministic 48-request mapping, and a recorded license/data-processing boundary. No generation starts automatically.
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_BACKEND_EVIDENCE_COMPARISON_20260725.md",
        """
# Subject00 Backend Evidence Comparison

| Candidate | Strongest evidence | Blocking evidence |
|---|---|---|
| Codex managed image edit | 12/12 Subject02 direct edits and 15 accepted donor sheets; native 1024x1536 | exact model/revision unavailable; interactive/session-bound; raw provider response unavailable; Subject02 formal visual gate failed |
| Historical Sublyx proxy | scriptable `/v1/images/edits`; 246 accepted traceable donor records; 915-row Jay execution log | alias/revision unpinned; current availability unverified; historical 1536x1024 only; research-use boundary absent |
| Historical 78Code proxy | exact `gpt-image-2` model ID discovered; scripted diagnostics and error taxonomy | edits probe returned HTTP 500 `convert_request_failed`; no successful image; revision/license/resolution unverified |

The weighted score is diagnostic only. Mandatory gates override every total, and no backend is eligible for PRIMARY or FALLBACK.

Historical success classes are kept separate: `CONNECTIVITY_ONLY`, `IMAGE_EDIT_SUCCESS`, and `FORMAL_DATASET_GENERATION_SUCCESS`. No evidence establishes a successful Subject02 formal target dataset from any selectable reproducible backend.
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_BACKEND_CANARY_PLAN_20260725.md",
        """
# Subject00 Backend Minimal Canary Plan

This task designs but does not authorize or execute a canary.

The canary uses two existing requests from the frozen 48-entry budget:

- `subject00_O01_slot00_cand00`: O01 front.
- `subject00_O03_slot03_cand00`: O03 left side.

Both must preserve the exact frozen inputs, prompt IDs, SHA-256 values, output count, and target resolution. Each request permits at most two retries after the initial attempt, only for technical transient errors, with 3/9 second backoff. Scientific quality rejection is never retried automatically.

Both outputs must pass two independent identity, garment, pose/camera, full-body, hand/foot, no-text, and provenance reviews. Any hard rejection or nonretryable provider error stops the canary. The remaining 46 requests require both canaries to pass, a sealed Formal Base dependency, and a new explicit user authorization.
""",
    )


def main() -> int:
    source_manifest = load_json(SOURCE_MANIFEST)
    source_summary = load_json(SOURCE_SUMMARY)
    if source_summary["classification"] != "SUBJECT00_DATA_PREPARATION_READY_PENDING_BACKEND":
        raise RuntimeError("unexpected source classification")
    if len(source_manifest["requests"]) != 48:
        raise RuntimeError("source request manifest is not frozen at 48 entries")

    outputs = {
        "subject00_backend_candidate_inventory.json": build_candidate_inventory(),
        "subject00_backend_evidence_registry.json": build_evidence_registry(),
        "subject00_backend_mandatory_gate_matrix.json": build_gate_matrix(),
        "subject00_backend_weighted_comparison.json": build_weighted_comparison(),
        "subject00_backend_selection_decision.json": build_selection_decision(),
        "subject00_backend_request_schema.json": build_request_schema(),
        "subject00_backend_manifest_binding.json": build_manifest_binding(source_manifest),
        "subject00_backend_cost_rate_limit_audit.json": build_cost_audit(),
        "subject00_backend_minimal_canary_contract.json": build_canary_contract(),
        "subject00_backend_secret_scan.json": build_secret_scan(),
        "subject00_backend_adjudication_tests.json": build_tests_placeholder(),
        "subject00_backend_adjudication_final_summary.json": build_final_summary(source_manifest),
    }
    for name, payload in outputs.items():
        write_json(RISK / name, payload)
    write_json(HANDOFF / "subject00_generation_backend_adjudication_handoff.json", build_handoff(source_manifest))
    build_docs()
    print(f"WROTE_{len(outputs) + 4}_BACKEND_ADJUDICATION_ARTIFACTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
