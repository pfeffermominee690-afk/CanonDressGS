#!/usr/bin/env python3
"""Build a compact repo index for a full external visual-asset registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-registry", type=Path, required=True)
    parser.add_argument("--transform-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    full = load(args.full_registry)
    transforms = load(args.transform_registry)
    selected_ids = sorted({asset_id for item in transforms["transforms"] for asset_id in item["asset_ids"]})
    by_id = {asset["asset_id"]: asset for asset in full["assets"]}
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise SystemExit(f"contact-sheet assets absent from full registry: {missing}")
    digest = hashlib.sha256()
    for asset in sorted(full["assets"], key=lambda item: item["asset_id"]):
        digest.update(f"{asset['asset_id']}\t{asset['original_sha256']}\n".encode("ascii"))
    result: Dict[str, Any] = {
        "schema_version": "paper_figure_asset_registry.repo_index.v1",
        "task_id": full["task_id"],
        "external_full_registry": {
            "path": str(args.full_registry),
            "bytes": args.full_registry.stat().st_size,
            "sha256": sha(args.full_registry),
            "asset_id_original_sha256_manifest_sha256": digest.hexdigest(),
        },
        "statistics": full["statistics"],
        "sources": full["sources"],
        "active_run_disposition": full["active_run_disposition"],
        "active_run_files_consumed": full["active_run_files_consumed"],
        "avatarrex_media_exports": full["avatarrex_media_exports"],
        "duplicate_sha_group_count": len(full["duplicate_sha_groups"]),
        "contact_sheet_asset_count": len(selected_ids),
        "contact_sheet_assets": [by_id[asset_id] for asset_id in selected_ids],
        "scope_note": "All assets are indexed in the external registry; full provenance is embedded here for every repo contact-sheet asset.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"contact_sheet_assets": len(selected_ids), "full_assets": len(full["assets"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
