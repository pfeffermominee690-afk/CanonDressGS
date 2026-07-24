#!/usr/bin/env python3
"""Summarize a full evidence-bank registry without reading source media."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.registry.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    assets = registry["assets"]
    by_source = Counter(asset["source_category"] for asset in assets)
    by_output = Counter(asset["output_root"] for asset in assets)
    by_eligibility = Counter(asset["paper_eligibility"] for asset in assets)
    by_type = Counter(asset["asset_type"] for asset in assets)
    historical = [asset for asset in assets if asset["source_category"] == "HISTORICAL_51_RUN"]
    result: Dict[str, Any] = {
        "schema_version": "paper_figure_evidence_statistics.v1",
        "task_id": registry["task_id"],
        "full_registry_path": str(args.registry),
        "global": registry["statistics"],
        "asset_type_counts": dict(sorted(by_type.items())),
        "eligibility_counts": dict(sorted(by_eligibility.items())),
        "source_category_counts": dict(sorted(by_source.items())),
        "output_root_counts": dict(sorted(by_output.items())),
        "requested_category_counts": {
            "historical_51_run_assets": len(historical),
            "rank_ablation_assets": sum("PAPER-A1-" in asset["experiment_id"] for asset in historical),
            "reference_count_assets": sum("PAPER-A7-" in asset["experiment_id"] for asset in historical),
            "feature_ablation_assets": sum(
                any(f"PAPER-A{index}-" in asset["experiment_id"] for index in range(2, 7))
                for asset in historical
            ),
            "historical_51_o07_assets": sum("O07" in asset["original_relative_path"].upper() for asset in historical),
            "o07_limitation_assets_all_audited": sum(
                "O07" in asset["original_relative_path"].upper() for asset in assets
            ),
            "representation_decoder_history_assets": by_source["METHOD_DEVELOPMENT_FAILURE_TRIAGE"],
            "geometry_causal_assets": by_source["GEOMETRY_CAUSAL_ATTRIBUTION"],
            "dual_support_assets": by_source["DUAL_SUPPORT_ALL_PAIR"],
            "controller_diagnostic_assets": by_source["CONTROLLER_DIAGNOSTIC"],
            "subject00_assets": by_source["SUBJECT00_BASE_AVATAR"],
        },
        "source_audits": registry["sources"],
        "active_run_files_consumed": registry["active_run_files_consumed"],
        "avatarrex_media_exports": registry["avatarrex_media_exports"],
        "paper_final": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["requested_category_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
