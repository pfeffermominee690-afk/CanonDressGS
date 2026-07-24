#!/usr/bin/env python3
"""Recompute path-inferred provenance fields without reading source media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from inventory_existing_visuals import infer_fields


INFERRED_FIELDS = (
    "attempt_id", "experiment_id", "method", "baseline_or_ablation", "identity",
    "garment", "garment_pair", "reference_set", "pose", "camera", "split",
    "rotation", "seed", "checkpoint_step", "checkpoint_sha256", "renderer_sha256",
    "renderer_config_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    registry = load(args.registry)
    spec = load(args.spec)
    by_root: Dict[str, Dict[str, Any]] = {str(item["root"]): item for item in spec["sources"]}
    changed_assets = 0
    changed_fields = 0
    for asset in registry["assets"]:
        source = by_root[asset["output_root"]]
        inferred = infer_fields(asset["original_relative_path"], source)
        asset_changed = False
        for field in INFERRED_FIELDS:
            if asset.get(field) != inferred[field]:
                asset[field] = inferred[field]
                changed_fields += 1
                asset_changed = True
        if asset_changed:
            changed_assets += 1
    registry["schema_version"] = "paper_figure_asset_registry.v2"
    registry["inference_repair"] = {
        "changed_assets": changed_assets,
        "changed_fields": changed_fields,
        "media_files_read": 0,
        "reason": "GARMENT_TOKEN_BOUNDARY_ACCEPTS_UNDERSCORE_DELIMITERS",
        "supersedes_registry": str(args.registry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(registry["inference_repair"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
