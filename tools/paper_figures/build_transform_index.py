#!/usr/bin/env python3
"""Combine contact-sheet and metric-plot transforms into one repo registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-transforms", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--metric-manifest", type=Path, required=True)
    parser.add_argument("--plot-root", type=Path, required=True)
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
    contact = load(args.contact_transforms)["transforms"]
    metric = load(args.metric_manifest)["plots"]
    transforms: List[Dict[str, Any]] = []
    for item in contact:
        matches = sorted(args.contact_root.rglob(Path(item["output_path"]).name))
        if len(matches) != 1:
            raise SystemExit(f"expected one repo contact sheet for {item['output_path']}, found {matches}")
        actual = sha(matches[0])
        if actual != item["output_sha256"]:
            raise SystemExit(f"contact sheet hash mismatch for {matches[0]}")
        copied = dict(item)
        copied["repo_output_path"] = str(matches[0])
        copied["repo_output_sha256"] = actual
        transforms.append(copied)
    for plot in metric:
        plot_id = plot["plot_id"]
        outputs: Dict[str, Dict[str, str]] = {}
        for suffix in ("png", "svg", "pdf", "source_json"):
            extension = "source.json" if suffix == "source_json" else suffix
            matches = sorted(args.plot_root.rglob(f"{plot_id}.{extension}"))
            if len(matches) != 1:
                raise SystemExit(f"expected one repo output for {plot_id}.{extension}, found {matches}")
            actual = sha(matches[0])
            expected = plot["outputs"][suffix]
            if actual != expected:
                raise SystemExit(f"plot hash mismatch for {matches[0]}: {actual} != {expected}")
            outputs[suffix] = {"path": str(matches[0]), "sha256": actual}
        transforms.append({
            "transform_id": f"METRIC-PLOT-{plot_id.upper().replace('_', '-')}",
            "operation": "DETERMINISTIC_MATPLOTLIB_FROM_STRUCTURED_JSON",
            "source_files": plot["source_files"],
            "data_origin": plot["data_origin"],
            "claim_status": plot["claim_status"],
            "outputs": outputs,
            "crop": None,
            "method_specific_enhancement": False,
            "ai_generated": False,
        })
    result = {
        "schema_version": "paper_figure_transform_registry.v2",
        "transforms": sorted(transforms, key=lambda item: item["transform_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"contact_transforms": len(contact), "metric_transforms": len(metric)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
