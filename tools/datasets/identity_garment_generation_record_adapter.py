#!/usr/bin/env python3
"""Validate and normalize a future garment generation record without submitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GARMENT_IDS = {"O01", "O02", "O03", "O04", "O08"}
IDENTITY_IDS = {"subject00", "avatarrex_lbn1"}
SEMANTIC_POSE_SLOTS = {
    "front",
    "front_left_three_quarter",
    "left",
    "back_left_three_quarter",
    "back",
    "back_right_three_quarter",
    "right",
    "front_right_three_quarter",
}
SHA_FIELDS = {
    "source_image_sha256",
    "source_mask_sha256",
    "prompt_sha256",
    "negative_prompt_sha256",
}


def derive_seed(identity_id: str, garment_id: str, semantic_pose_slot: str, attempt_index: int) -> int:
    key = f"{identity_id}|{garment_id}|{semantic_pose_slot}|{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def validate_and_normalize(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("identity_id") not in IDENTITY_IDS:
        raise ValueError("unsupported identity_id")
    if record.get("garment_id") not in GARMENT_IDS:
        raise ValueError("unsupported garment_id")
    if record.get("semantic_pose_slot") not in SEMANTIC_POSE_SLOTS:
        raise ValueError("unsupported semantic_pose_slot")
    attempt_index = record.get("attempt_index")
    if not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    for field in SHA_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{field} must be a lowercase SHA256")
    for field in ("provider", "model_id", "api_base"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{field} must be pinned before preparation")
    if not isinstance(record.get("image_parameters"), dict) or not record["image_parameters"]:
        raise ValueError("image_parameters must be complete and non-empty")

    normalized = dict(record)
    normalized["schema_version"] = "multi_identity.generation_event.v1"
    normalized["event_type"] = "REQUEST_PREPARED"
    normalized["seed"] = derive_seed(
        normalized["identity_id"],
        normalized["garment_id"],
        normalized["semantic_pose_slot"],
        attempt_index,
    )
    normalized["request_timestamp"] = None
    normalized["response_metadata"] = None
    normalized["output_sha256"] = None
    normalized["retry_reason"] = None
    normalized["moderation_or_error_state"] = None
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    normalized = validate_and_normalize(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event_type": "REQUEST_PREPARED", "request_submitted": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
