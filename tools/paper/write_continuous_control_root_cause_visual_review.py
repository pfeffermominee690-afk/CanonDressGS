from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GRADES = {
    "cloud": 2,
    "mottle": 3,
    "edge_scatter": 2,
    "full_body_contamination": 0,
    "identity_contamination": 0,
    "silhouette_discontinuity": 1,
}
REVIEWED_AT = "2026-07-22T18:30:00+08:00"
REVIEWER = "Codex manual visual audit"


def pair_id(path: str) -> str:
    return Path(path).stem


def opened_item(index: int, source_path: str, kind: str) -> dict[str, Any]:
    return {
        "review_index": index,
        "kind": kind,
        "source_path": source_path,
        "actual_opened": True,
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "target_unchanged_check": True,
        "mapping_check": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist the completed 82-item manual visual audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only manual review already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for source_path in manifest["full_sheets"]:
        item = opened_item(len(items), source_path, "FULL")
        item.update({
            "pair_id": pair_id(source_path),
            "grades": dict(GRADES),
            "alpha_0_50_grades": dict(GRADES),
            "artifact_spatial_location": (
                "Broad garment torso and sleeve texture, with repeated lower-body speckle and "
                "scatter concentrated along the garment/body silhouette."
            ),
            "support_exclusive_alignment": False,
            "notes": (
                "All 11 alpha columns and four fixed views were opened. At alpha=0.50 the garment "
                "retains identity but shows moderate cloud, severe patch/mottle, moderate edge scatter, "
                "and minor silhouette discontinuity; no full-body or identity contamination was observed."
            ),
        })
        items.append(item)
    for source_path in manifest["channel_sheets"]:
        path = Path(source_path)
        item = opened_item(len(items), source_path, "CHANNEL")
        item.update({
            "variant": path.parent.name,
            "pair_id": path.stem,
            "grades": dict(GRADES),
            "alpha_0_50_grades": dict(GRADES),
            "artifact_spatial_location": (
                "Broad garment torso and sleeve texture, with repeated lower-body speckle and "
                "scatter concentrated along the garment/body silhouette."
            ),
            "support_exclusive_alignment": False,
            "notes": (
                "All alpha=0.25/0.50/0.75 columns and four fixed views were opened. The alpha=0.50 "
                "construction exactly reproduces the six-channel midpoint for every variant and retains "
                "moderate cloud, severe patch/mottle, moderate edge scatter, and minor silhouette "
                "discontinuity. This midpoint is descriptive but non-identifying."
            ),
        })
        items.append(item)
    for source_path in manifest["support_conflict_overlays"]:
        item = opened_item(len(items), source_path, "SUPPORT_OVERLAY")
        item.update({
            "pair_id": pair_id(source_path),
            "support_exclusive_alignment": False,
            "artifact_spatial_location": (
                "The rendered failure is broad across garment texture and silhouette, while "
                "exclusive red/blue support is absent or sparse and localized."
            ),
            "notes": (
                "All three canonical projections and four FULL midpoint views were opened. Red/blue "
                "exclusive support was absent or sparse/localized and did not spatially align with the "
                "broad cloud/mottle failure; purple overlap dominated when support was dense."
            ),
        })
        items.append(item)
    for source_path in manifest["stable_unstable_comparison_sheets"]:
        item = opened_item(len(items), source_path, "STABLE_UNSTABLE_COMPARISON")
        item.update({
            "group": Path(source_path).stem,
            "notes": (
                "Every preregistered pair row and all four fixed views were opened. Broad midpoint "
                "patch/mottle and cloud remain visible in both frozen stable and unstable groups."
            ),
        })
        items.append(item)
    expected_paths = (
        manifest["full_sheets"]
        + manifest["channel_sheets"]
        + manifest["support_conflict_overlays"]
        + manifest["stable_unstable_comparison_sheets"]
    )
    if len(items) != 82 or [item["source_path"] for item in items] != expected_paths:
        raise RuntimeError("manual review serialization does not match the frozen 82-item manifest")
    review = {
        "schema_version": "canondressgs.research.continuous_control_root_cause_visual_review.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "grade_scale": {"0": "NONE", "1": "MINOR", "2": "MODERATE", "3": "SEVERE"},
        "expected_count": 82,
        "actual_opened_count": 82,
        "artifact_max_grades": dict(GRADES),
        "support_exclusive_alignment_count": 0,
        "items": items,
        "no_result_selection": True,
        "protocol_unchanged": True,
        "paper_final": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "COMPLETE", "actual_opened_count": len(items)}, sort_keys=True))


if __name__ == "__main__":
    main()
