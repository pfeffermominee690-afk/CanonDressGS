#!/usr/bin/env python3
"""Validate figure statuses and caption claims against the frozen boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_STATUSES = {
    "READY_FROM_HISTORICAL_EVIDENCE", "PENDING_PURE_ENDPOINT_CROSSFIT",
    "PENDING_COEFFICIENT_HEADROOM", "PENDING_LOO_ADAPTATION",
    "METHOD_FREEZE_PENDING_HEADROOM_AND_LOO", "SUPPLEMENTARY_EXTENSION_CANDIDATE",
    "SUPPLEMENTARY_DIAGNOSTIC_ONLY", "SUPPLEMENTARY_PORTABILITY_CANDIDATE",
    "HISTORICAL_LIMITATION_ONLY", "LICENSE_RESTRICTED_DO_NOT_EXPORT",
    "UNKNOWN_PROVENANCE_DO_NOT_USE", "REQUIRES_MANUAL_ADJUDICATION",
}
BANNED_STATUSES = {"FINAL", "PAPER_FINAL", "CAMERA_READY"}
FORBIDDEN_PHRASES = {
    "generalizable", "arbitrary garment", "unseen garment generalization",
    "novel-view generalization", "cross-identity garment editing",
    "superior to hard lookup", "continuous control", "teacher upper bound",
    "state of the art",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-registry", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    with args.claim_registry.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    failures: List[Dict[str, Any]] = []
    for figure in registry["figures"]:
        status = figure["status"]
        if status not in ALLOWED_STATUSES or status in BANNED_STATUSES:
            failures.append({"figure_id": figure["figure_id"], "reason": "INVALID_STATUS", "status": status})
        for claim in figure.get("claims", []):
            lowered = claim.lower()
            matches = sorted(phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered)
            if matches:
                failures.append({"figure_id": figure["figure_id"], "reason": "FORBIDDEN_CLAIM", "matches": matches})
    report = {
        "schema_version": "paper_figure_claim_boundary_verification.v1",
        "status": "PASS" if not failures else "FAIL",
        "figures_checked": len(registry["figures"]),
        "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
