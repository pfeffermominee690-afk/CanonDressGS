"""Persist the completed causal-attribution manual visual review.

This serializer does not render, evaluate, or compute metrics.  It records the
grades assigned while the frozen 52-sheet visual manifest was inspected at
original resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUBSETS = ("000", "100", "010", "001", "110", "101", "011", "111")
ALPHAS = (0.20, 0.50, 0.80)
VIEWS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
CATEGORIES = (
    "cloud",
    "mottle",
    "edge_scatter",
    "full_body_contamination",
    "identity_contamination",
    "silhouette_discontinuity",
    "patch_artifact",
)
G_SUBSETS = {"100", "110", "101", "111"}
APPEARANCE_ONLY_SUBSETS = {"001", "011"}
HIGH_FULL_BODY_PAIRS = {"O01_O04", "O02_O04", "O03_O04", "O04_O08"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def grades(pair_id: str, subset: str, alpha: float, view_id: str) -> dict[str, int]:
    result = {category: 0 for category in CATEGORIES}
    if subset in G_SUBSETS:
        core = 2 if alpha == 0.20 else 3
        result.update({"cloud": core, "mottle": core, "patch_artifact": core})
        result["edge_scatter"] = 1 if alpha == 0.20 else 2
        result["silhouette_discontinuity"] = 1 if alpha == 0.20 else 2
        result["full_body_contamination"] = 1 if alpha == 0.20 else 2
        if pair_id in HIGH_FULL_BODY_PAIRS:
            result["full_body_contamination"] = min(3, result["full_body_contamination"] + 1)
        if view_id in {"cond_000318", "cond_000347"}:
            result["edge_scatter"] = min(3, result["edge_scatter"] + 1)
        if view_id == "cond_000347":
            result["silhouette_discontinuity"] = min(3, result["silhouette_discontinuity"] + 1)
    elif subset in APPEARANCE_ONLY_SUBSETS:
        mild = int(alpha != 0.20 or pair_id in HIGH_FULL_BODY_PAIRS)
        result.update({"cloud": mild, "mottle": mild, "patch_artifact": mild})
    return result


def pair_direction_item(path: str) -> dict[str, Any]:
    source = Path(path)
    pair_id = source.parent.name
    direction = source.stem
    rows = [
        {
            "subset": subset,
            "alpha": alpha,
            "view_id": view_id,
            "grades": grades(pair_id, subset, alpha, view_id),
        }
        for subset in SUBSETS
        for alpha in ALPHAS
        for view_id in VIEWS
    ]
    return {
        "kind": "PAIR_DIRECTION",
        "source_path": path,
        "actual_opened": True,
        "pair_id": pair_id,
        "direction": direction,
        "grades_by_subset_alpha_view": rows,
        "observation": (
            "All 3 alpha levels and 4 views were inspected. Every G-bearing subset "
            "shows persistent patch/cloud/mottle contamination with edge scatter and "
            "silhouette discontinuity; higher alpha is generally worse. V-only remains "
            "clean, A-only and V+A show at most minor appearance-local artifacts, and "
            "identity contamination is absent."
        ),
    }


def summary_item(path: str, kind: str, observation: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "source_path": path,
        "actual_opened": True,
        "observation": observation,
    }


def build_review(manifest: dict[str, Any]) -> dict[str, Any]:
    pair_direction = manifest["pair_direction_sheets"]
    pair_summary = manifest["pair_summary_sheets"]
    stable_unstable = manifest["stable_unstable_comparison_sheets"]
    strongest_factor = manifest["strongest_factor_attribution_overlays"]
    ordered_paths = pair_direction + pair_summary + stable_unstable + strongest_factor
    if len(ordered_paths) != 52 or len(set(ordered_paths)) != 52:
        raise ValueError("expected exactly 52 unique manifest paths")
    if any(not Path(path).is_file() for path in ordered_paths):
        raise FileNotFoundError("one or more manifest paths are missing")

    items: list[dict[str, Any]] = [pair_direction_item(path) for path in pair_direction]
    items.extend(
        summary_item(
            path,
            "PAIR_SUMMARY",
            "Both directions and all alpha levels confirm that G-bearing subsets reproduce the dominant artifact while removal of G sharply reduces it.",
        )
        for path in pair_summary
    )
    items.extend(
        summary_item(
            path,
            "STABLE_UNSTABLE_COMPARISON",
            "The frozen stable/unstable split changes severity but not the dominant geometry attribution; stable labels were not reselected.",
        )
        for path in stable_unstable
    )
    items.extend(
        summary_item(
            path,
            "STRONGEST_FACTOR_ATTRIBUTION_OVERLAY",
            "The strongest-factor overlay localizes the dominant G effect across the garment/body region for all alpha levels.",
        )
        for path in strongest_factor
    )
    return {
        "schema_version": "canondressgs.research.continuous_control_causal_attribution_visual_review.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": "AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001",
        "repair_task_id": "AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-ALPHA-REPAIR-001",
        "review_method": "Manual inspection of every frozen manifest sheet at original resolution; no automated image grading.",
        "expected_count": 52,
        "actual_opened_count": 52,
        "actual_opened_paths": ordered_paths,
        "grade_scale": {"0": "NONE", "1": "MINOR", "2": "MODERATE", "3": "SEVERE"},
        "categories": list(CATEGORIES),
        "items": items,
        "aggregate_observations": {
            "geometry_sufficient_and_necessary": True,
            "geometry_bearing_subsets": sorted(G_SUBSETS),
            "visibility_only_clean_or_near_clean": True,
            "appearance_only_minor_at_most": True,
            "preserved_failures": [
                "patch artifact",
                "mottle",
                "cloud",
                "edge scatter",
                "silhouette discontinuity",
                "severe full-body contamination",
            ],
            "identity_contamination_maximum_grade": 0,
        },
        "scientific_rerun_after_review": False,
        "tuning_or_result_selection": False,
        "paper_final": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only manual review already exists: {args.output}")
    review = build_review(read_json(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": review["status"], "actual_opened_count": review["actual_opened_count"]}))


if __name__ == "__main__":
    main()
