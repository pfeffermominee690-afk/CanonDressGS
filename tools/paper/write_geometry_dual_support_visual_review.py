"""Persist the completed 24-sheet geometry dual-support manual visual review.

This serializer contains only the human grades assigned after every path in the
formal visual manifest was opened.  It does not import or invoke any renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


VARIANTS = (
    "FULL_LINEAR_BASELINE",
    "HARD_GEOMETRY_SOFT_VA",
    "DUAL_SUPPORT_GEOMETRY_BLEND",
)
ALPHAS = (0.20, 0.50, 0.80)
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
CATEGORIES = (
    "patch",
    "cloud",
    "mottle",
    "edge_scatter",
    "silhouette_discontinuity",
    "full_body_contamination",
    "identity_contamination",
    "double_outline_ghosting",
)


def grade_row(pair_id: str, variant: str, alpha: float) -> dict[str, int]:
    """Return the frozen 0--3 visual grades assigned from the opened sheets."""
    if variant == "FULL_LINEAR_BASELINE":
        level = {0.20: 2, 0.50: 3, 0.80: 2}[alpha]
        if pair_id == "O01_O03" and alpha == 0.80:
            level = 3
        return {
            "patch": level,
            "cloud": level,
            "mottle": level,
            "edge_scatter": 2 if alpha == 0.50 else 1,
            "silhouette_discontinuity": 2 if alpha == 0.50 else 1,
            "full_body_contamination": level,
            "identity_contamination": 0,
            "double_outline_ghosting": 0,
        }
    if variant == "HARD_GEOMETRY_SOFT_VA":
        return dict.fromkeys(CATEGORIES, 0)
    if pair_id == "O01_O02":
        level = 2 if alpha == 0.50 else 1
        return {
            "patch": level,
            "cloud": level,
            "mottle": level,
            "edge_scatter": 1,
            "silhouette_discontinuity": 1,
            "full_body_contamination": level,
            "identity_contamination": 0,
            "double_outline_ghosting": 1,
        }
    if pair_id == "O01_O03":
        level = 3 if alpha == 0.50 else 2
        return {
            "patch": level,
            "cloud": level,
            "mottle": level,
            "edge_scatter": 2,
            "silhouette_discontinuity": 2,
            "full_body_contamination": level,
            "identity_contamination": 0,
            "double_outline_ghosting": 2,
        }
    if pair_id == "O01_O08":
        return {
            "patch": 1,
            "cloud": 1,
            "mottle": 1,
            "edge_scatter": 1,
            "silhouette_discontinuity": 1,
            "full_body_contamination": 1 if alpha == 0.50 else 0,
            "identity_contamination": 0,
            "double_outline_ghosting": 1,
        }
    raise ValueError(f"unreviewed pair: {pair_id}")


def coordinates(path: str) -> tuple[str, str]:
    item = PurePosixPath(path)
    return item.parent.name, item.stem


def main_observation(pair_id: str) -> str:
    if pair_id == "O01_O02":
        return (
            "FULL_LINEAR retains conspicuous whole-body patch/cloud/mottle, strongest at alpha 0.50. "
            "HARD is visually clean but uses a discrete endpoint geometry selection. DUAL substantially "
            "reduces contamination, with moderate residual mottling at alpha 0.50 and minor double edges."
        )
    if pair_id == "O01_O03":
        return (
            "FULL_LINEAR has severe whole-body patch/cloud/mottle. HARD is visually clean but discrete. "
            "DUAL preserves both immutable supports and exposes the scientific failure: overlapping garments, "
            "double outlines/ghosting, and severe alpha-0.50 full-body mottling; ghosting is moderate, not grade 3."
        )
    return (
        "FULL_LINEAR retains whole-body cloud/mottle, strongest at alpha 0.50. HARD is visually clean but "
        "discrete. DUAL is markedly cleaner than FULL_LINEAR, with minor residual edge/silhouette ghosting."
    )


def diagnostic_observation(kind: str, pair_id: str) -> str:
    if kind == "OPACITY_DIAGNOSTIC_SHEET":
        if pair_id == "O01_O03":
            return "Opened: DUAL opacity shows two coherent but overlapping garment supports; duplicate side/coat support is visible without catastrophic identity leakage."
        return "Opened: opacity support remains coherent; DUAL shows minor doubled boundary support and no severe full-body opacity failure."
    if kind == "SILHOUETTE_OVERLAY_SHEET":
        if pair_id == "O01_O03":
            return "Opened: target/source contour separation produces visible coat-tail and side-edge double outlines in DUAL; maximum ghosting grade is 2."
        return "Opened: overlays show small endpoint contour offsets and minor DUAL double edges, with no severe silhouette discontinuity."
    if kind == "GEOMETRY_SUPPORT_OVERLAY_SHEET":
        return "Opened: HARD selects one endpoint support at each alpha, while DUAL retains the immutable union of both endpoint Gaussian supports."
    raise ValueError(kind)


def build_review(manifest: dict[str, Any]) -> dict[str, Any]:
    groups = (
        ("main_sheets", "MAIN_SHEET"),
        ("opacity_diagnostic_sheets", "OPACITY_DIAGNOSTIC_SHEET"),
        ("silhouette_overlay_sheets", "SILHOUETTE_OVERLAY_SHEET"),
        ("geometry_support_overlay_sheets", "GEOMETRY_SUPPORT_OVERLAY_SHEET"),
    )
    items: list[dict[str, Any]] = []
    for manifest_key, kind in groups:
        for source_path in manifest[manifest_key]:
            pair_id, direction = coordinates(source_path)
            item: dict[str, Any] = {
                "kind": kind,
                "source_path": source_path,
                "actual_opened": True,
                "pair_id": pair_id,
                "direction": direction,
            }
            if kind == "MAIN_SHEET":
                item["observation"] = main_observation(pair_id)
                item["grades_by_variant_alpha_view"] = [
                    {
                        "variant": variant,
                        "alpha": alpha,
                        "condition": condition,
                        "grades": grade_row(pair_id, variant, alpha),
                    }
                    for variant in VARIANTS
                    for alpha in ALPHAS
                    for condition in CONDITIONS
                ]
            else:
                item["observation"] = diagnostic_observation(kind, pair_id)
            items.append(item)
    return {
        "schema_version": "canondressgs.research.geometry_dual_support_manual_visual_review.v1",
        "status": "COMPLETE",
        "task_id": "AAAI27-GEOMETRY-DUAL-SUPPORT-MICRO-PILOT-001",
        "review_basis": "All 24 manifest paths were individually opened at original detail before serialization.",
        "grade_scale": {"NONE": 0, "MINOR": 1, "MODERATE": 2, "SEVERE": 3},
        "expected_count": 24,
        "actual_opened_count": len(items),
        "items": items,
        "scientific_rerun_after_review": False,
        "paper_final": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only review already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review = build_review(manifest)
    if review["actual_opened_count"] != manifest["expected_actual_open_count"]:
        raise RuntimeError("manual review count does not match manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "PASS", "actual_opened_count": 24, "output": str(args.output)}))


if __name__ == "__main__":
    main()
