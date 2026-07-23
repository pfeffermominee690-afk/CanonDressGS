#!/usr/bin/env python3
"""Offline request, provenance, review, and manifest primitives.

This module deliberately has no generation client and no connectivity probe.
It prepares reproducible envelopes and append-only local records only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


IDENTITIES = ("subject00", "avatarrex_lbn1")
GARMENTS = ("O01", "O02", "O03", "O04", "O08")
SLOTS = (
    "front",
    "front_left_three_quarter",
    "left",
    "back_left_three_quarter",
    "back",
    "back_right_three_quarter",
    "right",
    "front_right_three_quarter",
)
REFERENCE_SLOTS = ("front", "left", "back", "right")
INITIAL_DECISIONS = ("ACCEPT", "REJECT", "MAYBE")
REVIEW_DIMENSIONS = (
    "identity_preservation",
    "garment_semantics",
    "full_body",
    "hands",
    "feet",
    "anatomy",
    "background",
    "blur",
    "view",
    "cross_view_consistency",
    "body_conforming_bias",
    "garment_volume",
)
GROUP_DIMENSIONS = (
    "color_consistency",
    "material_consistency",
    "sleeve_length_consistency",
    "hem_consistency",
    "front_back_semantic_consistency",
    "left_right_consistency",
    "identity_consistency",
    "accessory_consistency",
)
RETRYABLE_ERROR_CLASSES = (
    "NETWORK",
    "TIMEOUT",
    "HTTP_5XX",
    "PROVIDER_TRANSIENT_ERROR",
)
TERMINAL_REQUEST_EVENTS = (
    "REQUEST_SUCCEEDED",
    "REQUEST_FAILED",
    "REQUEST_MODERATED",
)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|authorization|credential[_-]?value|secret)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    ("OPENAI_STYLE_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)),
    (
        "ASSIGNED_SECRET",
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|authorization|credential[_-]?value|secret)"
            r"\s*[:=]\s*[\"']?(?!null\b|none\b|redacted\b|unset\b|pending\b)"
            r"[A-Za-z0-9._~+/=-]{12,}"
        ),
    ),
)
_SAFE_TEST_SENTINELS = ("deliberately_fake_secret_value_123",)


class ContractError(ValueError):
    """Raised when an offline workflow contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_request_id(
    identity_id: str,
    garment_id: str,
    semantic_pose_slot: str,
    candidate_index: int,
    attempt_index: int,
) -> tuple[str, str]:
    _validate_coordinates(identity_id, garment_id, semantic_pose_slot, candidate_index)
    if attempt_index < 0:
        raise ContractError("attempt_index must be non-negative")
    logical = {
        "candidate_index": candidate_index,
        "garment_id": garment_id,
        "identity_id": identity_id,
        "semantic_pose_slot": semantic_pose_slot,
        "version": "multi_identity.request_id.v1",
    }
    parent = "mirp_" + hashlib.sha256(canonical_json(logical)).hexdigest()[:24]
    attempt = {**logical, "attempt_index": attempt_index}
    request = "mir_" + hashlib.sha256(canonical_json(attempt)).hexdigest()[:24]
    return request, parent


def candidate_filename(
    identity_id: str,
    garment_id: str,
    semantic_pose_slot: str,
    candidate_index: int,
    request_id: str,
) -> str:
    _validate_coordinates(identity_id, garment_id, semantic_pose_slot, candidate_index)
    if not re.fullmatch(r"mir_[0-9a-f]{24}", request_id):
        raise ContractError("invalid deterministic request_id")
    return (
        f"{identity_id}__{garment_id}__{semantic_pose_slot}__"
        f"candidate{candidate_index:02d}__{request_id}.png"
    )


def build_dry_run_request(
    *,
    identity_id: str,
    garment_id: str,
    semantic_pose_slot: str,
    candidate_index: int,
    garment_prompt: str,
    negative_prompt: str,
    source_root: str = "multi_garment_benchmark",
) -> dict[str, Any]:
    """Build a blocked request without reading credentials or touching the network."""
    _validate_coordinates(identity_id, garment_id, semantic_pose_slot, candidate_index)
    request_id, parent_request_id = deterministic_request_id(
        identity_id,
        garment_id,
        semantic_pose_slot,
        candidate_index,
        0,
    )
    base = PurePosixPath(source_root) / identity_id
    output_name = candidate_filename(
        identity_id,
        garment_id,
        semantic_pose_slot,
        candidate_index,
        request_id,
    )
    request: dict[str, Any] = {
        "schema_version": "multi_identity.generation_request.v1",
        "mode": "DRY_RUN_ONLY",
        "status": "BLOCKED_BACKEND_SELECTION_AND_FORMAL_BASE",
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "attempt_index": 0,
        "candidate_index": candidate_index,
        "identity_id": identity_id,
        "garment_id": garment_id,
        "semantic_pose_slot": semantic_pose_slot,
        "condition_id": f"{identity_id}_{semantic_pose_slot}_FORMAL_BASE_PENDING",
        "source_inputs": {
            "condition_rgb": {
                "path": str(base / "condition_manifests" / f"{semantic_pose_slot}_rgb.png"),
                "sha256": None,
                "status": "FORMAL_BASE_PENDING",
            },
            "condition_alpha_mask": {
                "path": str(base / "condition_manifests" / f"{semantic_pose_slot}_alpha.png"),
                "sha256": None,
                "status": "FORMAL_BASE_PENDING",
            },
            "identity_manifest": {
                "path": str(base / "identity_manifest" / "identity_manifest.json"),
                "sha256": None,
                "status": "FORMAL_BASE_PENDING",
            },
        },
        "prompt_sha256": sha256_text(garment_prompt),
        "negative_prompt_sha256": sha256_text(negative_prompt),
        "backend": {
            "provider": None,
            "model_id": None,
            "base_url_identifier": None,
            "endpoint": None,
            "selection_status": "GENERATION_BACKEND_SELECTION_REQUIRED",
        },
        "request_parameters": {
            "input_image_format": "PNG",
            "output_format": "PNG",
            "output_width": 1024,
            "output_height": 1536,
            "wire_size": None,
            "quality": None,
            "background": "LIGHT_GRAY_OR_WHITE",
            "n": 1,
            "timeout_seconds": 300,
            "moderation": "PROVIDER_NATIVE_PLUS_LOCAL_SCREENING_REQUIRED",
        },
        "retry_policy_id": "multi_identity.retry_contract.v1",
        "credential_env_name": "MULTI_IDENTITY_GENERATION_API_KEY",
        "reproducibility": {
            "level": "REQUEST_PROVENANCE_REPRODUCIBLE_ONLY",
            "seed_support": "UNVERIFIED_UNTIL_BACKEND_SELECTION",
        },
        "paths": {
            "output_path": str(base / "generated_candidates" / output_name),
            "candidate_manifest_path": str(
                base / "generated_candidates" / f"{output_name}.manifest.json"
            ),
            "request_event_path": str(base / "provenance" / "request_events.jsonl"),
            "retry_event_path": str(base / "provenance" / "request_events.jsonl"),
            "review_event_path": str(base / "screening" / "review_events.jsonl"),
        },
        "prepared_at": None,
        "network_call_performed": False,
    }
    validate_request(request)
    return request


def validate_request(request: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "mode",
        "status",
        "request_id",
        "parent_request_id",
        "attempt_index",
        "candidate_index",
        "identity_id",
        "garment_id",
        "semantic_pose_slot",
        "condition_id",
        "source_inputs",
        "prompt_sha256",
        "negative_prompt_sha256",
        "backend",
        "request_parameters",
        "retry_policy_id",
        "credential_env_name",
        "reproducibility",
        "paths",
        "prepared_at",
        "network_call_performed",
    }
    missing = required - set(request)
    if missing:
        raise ContractError(f"request missing fields: {sorted(missing)}")
    forbidden = {key for key in request if _SECRET_KEY.search(key) and key != "credential_env_name"}
    if forbidden:
        raise ContractError(f"request contains forbidden credential fields: {sorted(forbidden)}")
    _validate_coordinates(
        str(request["identity_id"]),
        str(request["garment_id"]),
        str(request["semantic_pose_slot"]),
        int(request["candidate_index"]),
    )
    expected_request, expected_parent = deterministic_request_id(
        str(request["identity_id"]),
        str(request["garment_id"]),
        str(request["semantic_pose_slot"]),
        int(request["candidate_index"]),
        int(request["attempt_index"]),
    )
    if request["request_id"] != expected_request or request["parent_request_id"] != expected_parent:
        raise ContractError("request ID does not match deterministic coordinates")
    for field in ("prompt_sha256", "negative_prompt_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(request[field])):
            raise ContractError(f"invalid {field}")
    for source in request["source_inputs"].values():
        if source["status"] == "FORMAL_BASE_PENDING" and source["sha256"] is not None:
            raise ContractError("pending source must not invent a SHA")
        if source["status"] == "READY_AND_HASHED" and not re.fullmatch(
            r"[0-9a-f]{64}", str(source["sha256"])
        ):
            raise ContractError("ready source must contain a valid SHA")
    seed_status = request["reproducibility"]["seed_support"]
    if seed_status != "SUPPORTED_AND_VERIFIED" and "seed" in request:
        raise ContractError("seed must be omitted unless support is verified")
    if request["network_call_performed"] is not False:
        raise ContractError("dry-run cannot record a network call")
    if request["mode"] == "AUTHORIZED_EXECUTION":
        if request["status"] != "EXECUTION_AUTHORIZED":
            raise ContractError("authorized mode requires execution authorization")
        if request["backend"]["selection_status"] != "GENERATION_BACKEND_CONTRACT_READY":
            raise ContractError("authorized execution requires a selected backend")
        if any(request["backend"][field] is None for field in ("provider", "model_id", "base_url_identifier", "endpoint")):
            raise ContractError("authorized execution requires complete backend coordinates")
        if any(source["status"] != "READY_AND_HASHED" for source in request["source_inputs"].values()):
            raise ContractError("authorized execution requires all formal source inputs")
        if request["request_parameters"]["wire_size"] is None or request["request_parameters"]["quality"] is None:
            raise ContractError("authorized execution requires complete wire parameters")
    _reject_secret_values(request)


def append_jsonl_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append a hash-chained event. Existing bytes are never rewritten."""
    _reject_secret_values(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash: str | None = None
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as stream:
            for raw_line in stream:
                if raw_line.strip():
                    previous = json.loads(raw_line)
                    previous_hash = previous.get("event_hash")
    payload = dict(event)
    payload.setdefault("recorded_at", utc_now())
    payload["previous_event_hash"] = previous_hash
    payload["event_hash"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    encoded = canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def validate_request_event_sequence(events: Sequence[Mapping[str, Any]]) -> None:
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_request.setdefault(str(event["request_id"]), []).append(event)
    for request_id, request_events in by_request.items():
        types = [event["event_type"] for event in request_events]
        if not types or types[0] != "REQUEST_PREPARED":
            raise ContractError(f"{request_id} must begin with REQUEST_PREPARED")
        terminals = [event for event in types if event in TERMINAL_REQUEST_EVENTS]
        if len(terminals) > 1:
            raise ContractError(f"{request_id} has multiple terminal events")
        if terminals and types[-1] not in TERMINAL_REQUEST_EVENTS:
            raise ContractError(f"{request_id} has an event after terminal state")


def validate_review_event(event: Mapping[str, Any], *, group: bool = False) -> None:
    required = {
        "candidate_id" if not group else "group_id",
        "reviewer_id",
        "review_stage",
        "decision",
        "scores",
        "comment",
        "timestamp",
    }
    missing = required - set(event)
    if missing:
        raise ContractError(f"review event missing fields: {sorted(missing)}")
    if event["review_stage"] not in ("INITIAL", "ADJUDICATION"):
        raise ContractError("unknown review_stage")
    if event["decision"] not in INITIAL_DECISIONS:
        raise ContractError("unknown decision")
    dimensions = GROUP_DIMENSIONS if group else REVIEW_DIMENSIONS
    if set(event["scores"]) != set(dimensions):
        raise ContractError("review scores must contain every independent dimension")
    allowed_scores = {"PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE"}
    if any(score not in allowed_scores for score in event["scores"].values()):
        raise ContractError("invalid review dimension score")
    _reject_secret_values(event)


def resolve_initial_decision(
    reviewer_a: Mapping[str, Any] | None,
    reviewer_b: Mapping[str, Any] | None,
) -> str:
    if reviewer_a is None or reviewer_b is None:
        return "PENDING_INDEPENDENT_REVIEW"
    decisions = (reviewer_a["decision"], reviewer_b["decision"])
    if any(decision == "REJECT" for decision in decisions):
        return "REJECT"
    if decisions == ("ACCEPT", "ACCEPT"):
        return "ACCEPT"
    return "ADJUDICATION_REQUIRED"


def resolve_adjudication(
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    adjudicator: Mapping[str, Any] | None,
) -> str:
    initial = resolve_initial_decision(reviewer_a, reviewer_b)
    if initial != "ADJUDICATION_REQUIRED":
        return initial
    if adjudicator is None:
        return initial
    if adjudicator.get("review_stage") != "ADJUDICATION":
        raise ContractError("third review must be an explicit adjudication")
    if adjudicator["decision"] == "ACCEPT":
        return "ACCEPT"
    if adjudicator["decision"] == "REJECT":
        return "REJECT"
    return "ADJUDICATION_REQUIRED"


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def verify_sealed_manifest(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("content_sha256")
    if not isinstance(expected, str):
        return False
    payload = dict(manifest)
    payload.pop("content_sha256")
    return hashlib.sha256(canonical_json(payload)).hexdigest() == expected


def seal_candidate_manifest(
    request: Mapping[str, Any],
    *,
    candidate_path: str,
    candidate_sha256: str,
    request_event_hash: str,
) -> dict[str, Any]:
    validate_request(request)
    if request["mode"] != "AUTHORIZED_EXECUTION" or request["status"] != "EXECUTION_AUTHORIZED":
        raise ContractError("blocked or dry-run request cannot produce a candidate manifest")
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
        raise ContractError("invalid candidate_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", request_event_hash):
        raise ContractError("invalid request_event_hash")
    expected_name = candidate_filename(
        str(request["identity_id"]),
        str(request["garment_id"]),
        str(request["semantic_pose_slot"]),
        int(request["candidate_index"]),
        str(request["request_id"]),
    )
    if PurePosixPath(candidate_path).name != expected_name:
        raise ContractError("candidate path does not match traceable naming contract")
    return seal_manifest(
        {
            "schema_version": "multi_identity.candidate_manifest.v1",
            "state": "CANDIDATE_SEALED",
            "candidate_id": PurePosixPath(candidate_path).stem,
            "candidate_path": candidate_path,
            "candidate_sha256": candidate_sha256,
            "request_id": request["request_id"],
            "parent_request_id": request["parent_request_id"],
            "request_event_hash": request_event_hash,
            "identity_id": request["identity_id"],
            "garment_id": request["garment_id"],
            "semantic_pose_slot": request["semantic_pose_slot"],
            "candidate_index": request["candidate_index"],
            "prompt_sha256": request["prompt_sha256"],
            "negative_prompt_sha256": request["negative_prompt_sha256"],
            "source_inputs": request["source_inputs"],
            "backend": request["backend"],
            "request_parameters": request["request_parameters"],
        }
    )


def seal_accepted_manifest(
    candidate_manifest: Mapping[str, Any],
    *,
    review_resolution_event_hash: str,
    group_resolution_event_hash: str,
    group_review_status: str,
) -> dict[str, Any]:
    _validate_candidate_manifest(candidate_manifest)
    if group_review_status != "PASS":
        raise ContractError("candidate cannot be accepted before group PASS")
    for value in (review_resolution_event_hash, group_resolution_event_hash):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ContractError("invalid review event hash")
    return seal_manifest(
        {
            "schema_version": "multi_identity.accepted_manifest.v1",
            "state": "ACCEPTED_REFERENCE",
            "candidate_manifest_sha256": candidate_manifest["content_sha256"],
            "candidate_path": candidate_manifest["candidate_path"],
            "candidate_sha256": candidate_manifest["candidate_sha256"],
            "identity_id": candidate_manifest["identity_id"],
            "garment_id": candidate_manifest["garment_id"],
            "semantic_pose_slot": candidate_manifest["semantic_pose_slot"],
            "review_resolution_event_hash": review_resolution_event_hash,
            "group_resolution_event_hash": group_resolution_event_hash,
            "group_review_status": group_review_status,
            "candidate_pixels_copied": False,
        }
    )


def seal_rejected_manifest(
    candidate_manifest: Mapping[str, Any],
    *,
    rejection_event_hash: str,
    rejection_class: str,
) -> dict[str, Any]:
    _validate_candidate_manifest(candidate_manifest)
    if not re.fullmatch(r"[0-9a-f]{64}", rejection_event_hash):
        raise ContractError("invalid rejection event hash")
    if not rejection_class:
        raise ContractError("rejection_class is required")
    return seal_manifest(
        {
            "schema_version": "multi_identity.rejected_manifest.v1",
            "state": "REJECTED",
            "candidate_manifest_sha256": candidate_manifest["content_sha256"],
            "candidate_path": candidate_manifest["candidate_path"],
            "candidate_sha256": candidate_manifest["candidate_sha256"],
            "identity_id": candidate_manifest["identity_id"],
            "garment_id": candidate_manifest["garment_id"],
            "semantic_pose_slot": candidate_manifest["semantic_pose_slot"],
            "rejection_event_hash": rejection_event_hash,
            "rejection_class": rejection_class,
            "candidate_retained": True,
        }
    )


def _validate_candidate_manifest(candidate_manifest: Mapping[str, Any]) -> None:
    if candidate_manifest.get("state") != "CANDIDATE_SEALED":
        raise ContractError("expected sealed candidate manifest")
    if not verify_sealed_manifest(candidate_manifest):
        raise ContractError("candidate manifest seal is invalid")


def evaluate_garment_acceptance(
    slot_records: Sequence[Mapping[str, Any]],
    group_review_status: str,
) -> dict[str, Any]:
    records = {str(record["semantic_pose_slot"]): record for record in slot_records}
    accepted_slots = {
        slot
        for slot, record in records.items()
        if record.get("decision") == "ACCEPT"
    }
    output_shas = [
        str(record.get("candidate_sha256"))
        for record in slot_records
        if record.get("candidate_sha256")
    ]
    hard_failures = {
        "identity_contamination": sum(
            int(record.get("identity_contamination", 0)) for record in slot_records
        ),
        "severe_anatomy_failure": sum(
            int(record.get("severe_anatomy_failure", 0)) for record in slot_records
        ),
        "text_or_watermark": sum(
            int(record.get("text_or_watermark", 0)) for record in slot_records
        ),
    }
    full_body_pass = all(record.get("full_body_completeness") == "PASS" for record in slot_records)
    passed = (
        accepted_slots == set(SLOTS)
        and set(REFERENCE_SLOTS).issubset(accepted_slots)
        and group_review_status == "PASS"
        and all(value == 0 for value in hard_failures.values())
        and full_body_pass
        and len(output_shas) == len(set(output_shas)) == len(SLOTS)
    )
    return {
        "status": "ACCEPT" if passed else "GARMENT_SLOT_GENERATION_INCOMPLETE",
        "endpoint_slots_accepted": len(accepted_slots & set(SLOTS)),
        "reference_slots_accepted": len(accepted_slots & set(REFERENCE_SLOTS)),
        "group_review_status": group_review_status,
        "hard_failures": hard_failures,
        "full_body_completeness": "PASS" if full_body_pass else "FAIL",
        "unique_candidate_sha_count": len(set(output_shas)),
    }


def scan_credential_leaks(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Return only path/rule identifiers; never return matching secret text."""
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for sentinel in _SAFE_TEST_SENTINELS:
            text = text.replace(sentinel, "SAFE")
        for rule, pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(text):
                findings.append({"path": str(path), "rule": rule})
    return findings


def assert_credential_safe(paths: Iterable[Path]) -> None:
    findings = scan_credential_leaks(paths)
    if findings:
        raise ContractError(f"API_CREDENTIAL_LEAK_RISK: {len(findings)} file/rule findings")


def _validate_coordinates(
    identity_id: str,
    garment_id: str,
    semantic_pose_slot: str,
    candidate_index: int,
) -> None:
    if identity_id not in IDENTITIES:
        raise ContractError("unknown identity_id")
    if garment_id not in GARMENTS:
        raise ContractError("unknown garment_id")
    if semantic_pose_slot not in SLOTS:
        raise ContractError("unknown semantic_pose_slot")
    if candidate_index < 0:
        raise ContractError("candidate_index must be non-negative")


def _reject_secret_values(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _SECRET_KEY.search(str(key)) and key != "credential_env_name":
                raise ContractError(f"forbidden credential key at {child_path}")
            _reject_secret_values(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_values(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for rule, pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                raise ContractError(f"{rule} detected at {path}")


if __name__ == "__main__":
    raise SystemExit(
        "This module is offline-only; import its builders from a validated workflow."
    )
