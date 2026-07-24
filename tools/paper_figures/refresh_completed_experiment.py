#!/usr/bin/env python3
"""Append a completed, sealed experiment to the evidence bank.

This interface never discovers active experiments. The caller must provide an
explicit final summary, seal marker, and completed registry. Without --execute
the tool performs validation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--completed-registry", type=Path, required=True)
    parser.add_argument("--target-registry", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    missing = [str(path) for path in (args.final_summary, args.seal, args.completed_registry) if not path.is_file()]
    if missing:
        raise SystemExit(f"refresh gates missing: {missing}")
    summary = load(args.final_summary)
    required = {
        "final_classification": bool(summary.get("final_classification") or summary.get("classification")),
        "final_head": bool(summary.get("final_head") or summary.get("head")),
        "attempt_complete": summary.get("attempt_complete") is True,
        "output_sealed": summary.get("output_sealed") is True,
    }
    if not all(required.values()):
        raise SystemExit(f"refresh gates not satisfied: {required}")
    completed = load(args.completed_registry)
    if not args.execute:
        print(json.dumps({"status": "VALIDATED_NOT_EXECUTED", "gates": required}, indent=2, sort_keys=True))
        return 0
    target: Dict[str, Any] = load(args.target_registry)
    existing = {asset["asset_id"]: asset for asset in target["assets"]}
    for asset in completed["assets"]:
        if asset["asset_id"] in existing and existing[asset["asset_id"]] != asset:
            raise SystemExit(f"append-only conflict for {asset['asset_id']}")
        existing[asset["asset_id"]] = asset
    target["assets"] = [existing[key] for key in sorted(existing)]
    args.target_registry.write_text(json.dumps(target, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "APPENDED", "asset_count": len(target["assets"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
