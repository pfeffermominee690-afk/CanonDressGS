#!/usr/bin/env python3
"""Verify paper-figure asset and transform provenance registries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REQUIRED_FIELDS = (
    "asset_id", "source_category", "original_absolute_path", "original_relative_path",
    "source_branch", "source_head", "output_root", "task_id", "attempt_id",
    "experiment_id", "method", "baseline_or_ablation", "identity", "garment",
    "garment_pair", "reference_set", "pose", "camera", "split", "rotation", "seed",
    "checkpoint_step", "checkpoint_sha256", "renderer_sha256", "renderer_config_sha256",
    "original_sha256", "width", "height", "channels", "crop", "resize",
    "composition_transform", "output_sha256", "claim_status", "license_status",
    "paper_eligibility", "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--transform-registry", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r} in {path}")
            value[key] = item
        return value

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject)


def main() -> int:
    args = parse_args()
    registry = load_json(args.registry)
    assets = registry.get("assets", registry.get("contact_sheet_assets", []))
    failures: List[Dict[str, Any]] = []
    asset_ids: List[str] = []
    for asset in assets:
        asset_ids.append(asset.get("asset_id", ""))
        missing = [field for field in REQUIRED_FIELDS if field not in asset]
        if missing:
            failures.append({"asset_id": asset.get("asset_id"), "reason": "MISSING_FIELDS", "fields": missing})
        complete = bool(asset.get("source_head") and asset.get("attempt_id") and asset.get("original_sha256"))
        if complete != bool(asset.get("provenance_complete")):
            failures.append({"asset_id": asset.get("asset_id"), "reason": "PROVENANCE_FLAG_MISMATCH"})
        if not complete and asset.get("paper_eligibility") != "UNKNOWN_PROVENANCE_DO_NOT_USE":
            failures.append({"asset_id": asset.get("asset_id"), "reason": "UNSUPPORTED_PROVENANCE_NOT_REJECTED"})
    duplicate_ids = sorted(key for key, count in Counter(asset_ids).items() if count > 1)
    if duplicate_ids:
        failures.append({"reason": "DUPLICATE_ASSET_IDS", "asset_ids": duplicate_ids})
    if registry.get("active_run_files_consumed") != 0:
        failures.append({"reason": "ACTIVE_RUN_FILES_CONSUMED_NONZERO"})
    if registry.get("avatarrex_media_exports") != 0:
        failures.append({"reason": "AVATARREX_MEDIA_EXPORT_NONZERO"})
    transforms = []
    if args.transform_registry:
        transforms = load_json(args.transform_registry).get("transforms", [])
        for transform in transforms:
            if transform.get("method_specific_enhancement"):
                failures.append({"reason": "METHOD_SPECIFIC_ENHANCEMENT", "transform": transform.get("transform_id")})
            if transform.get("ai_generated"):
                failures.append({"reason": "AI_GENERATED_TRANSFORM", "transform": transform.get("transform_id")})
    report = {
        "schema_version": "paper_figure_provenance_verification.v1",
        "status": "PASS" if not failures else "FAIL",
        "assets_checked": len(assets),
        "transforms_checked": len(transforms),
        "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
