#!/usr/bin/env python3
"""Validate a future Subject00 canary record without submitting a request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GARMENT_IDS = {"O01", "O03", "O04"}
SLOTS = {
    "front",
    "front_left_three_quarter",
    "front_right_three_quarter",
    "left",
    "right",
    "back_left_three_quarter",
    "back_right_three_quarter",
    "back",
}
SHA_FIELDS = {
    "source_rgb_sha256",
    "source_alpha_sha256",
    "prompt_sha256",
    "negative_prompt_sha256",
}


def derive_seed(garment_id: str, slot: str, attempt_index: int) -> int:
    key = f"subject00|{garment_id}|{slot}|{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def validate_and_normalize(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("identity_id") != "subject00":
        raise ValueError("identity_id must be subject00")
    garment_id = record.get("garment_id")
    slot = record.get("semantic_pose_slot")
    if garment_id not in GARMENT_IDS:
        raise ValueError("garment_id must be O01, O03, or O04")
    if slot not in SLOTS:
        raise ValueError("semantic_pose_slot is outside the frozen eight slots")
    attempt_index = record.get("attempt_index")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    for field in SHA_FIELDS:
        _sha256(record.get(field), field)
    for field in ("provider", "model_id", "base_url_identifier", "quality"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip() or "PENDING" in value:
            raise ValueError(f"{field} must be frozen before record preparation")
    image_size = record.get("image_size")
    if image_size != {"width": 1024, "height": 1536}:
        raise ValueError("image_size must match the frozen portrait contract")
    if attempt_index > 0 and not record.get("retry_parent_record_id"):
        raise ValueError("retry_parent_record_id is required for append-only retries")
    normalized = dict(record)
    normalized.update(
        {
            "schema_version": "subject00.three_garment.generation_event.v1",
            "event_type": "REQUEST_PREPARED",
            "seed": derive_seed(garment_id, slot, attempt_index),
            "request_timestamp": "NOT_SUBMITTED",
            "response_metadata": {},
            "output_sha256": "NOT_AVAILABLE_BEFORE_SUCCESS",
            "moderation_state": "NOT_SUBMITTED",
            "request_submitted": False,
        }
    )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    normalized = validate_and_normalize(
        json.loads(arguments.record.read_text(encoding="utf-8"))
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"event_type": "REQUEST_PREPARED", "request_submitted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
