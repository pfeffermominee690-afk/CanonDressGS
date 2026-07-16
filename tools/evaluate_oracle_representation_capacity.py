from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "rgb_l1", "psnr", "ssim", "lpips", "alpha_l1", "mask_iou",
    "clothing_rgb_l1", "clothing_lpips", "non_clothing_rgb_preservation",
    "non_clothing_alpha_preservation",
)


def evaluate_capacity(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    for method in ("base", "anchor", "gaussian"):
        missing = set(REQUIRED_METRICS).difference(metrics.get(method, {}))
        if missing:
            raise ValueError(f"{method} metrics missing: {sorted(missing)}")
    base, anchor, gaussian = metrics["base"], metrics["anchor"], metrics["gaussian"]

    def gaussian_pass() -> tuple[bool, dict[str, bool]]:
        checks = {
            "clothing_rgb_improvement_30pct": gaussian["clothing_rgb_l1"] <= 0.70 * base["clothing_rgb_l1"],
            "clothing_lpips_improvement_20pct": gaussian["clothing_lpips"] <= 0.80 * base["clothing_lpips"],
            "mask_iou": gaussian["mask_iou"] >= base["mask_iou"] + 0.02 or gaussian["mask_iou"] >= 0.95,
            "non_clothing_rgb_within_15pct": gaussian["non_clothing_rgb_preservation"] <= 1.15 * base["non_clothing_rgb_preservation"],
        }
        return all(checks.values()), checks

    gaussian_ok, gaussian_checks = gaussian_pass()
    anchor_checks = {
        "base_clothing_rgb_improvement_30pct": anchor["clothing_rgb_l1"] <= 0.70 * base["clothing_rgb_l1"],
        "base_clothing_lpips_improvement_20pct": anchor["clothing_lpips"] <= 0.80 * base["clothing_lpips"],
        "base_mask_iou": anchor["mask_iou"] >= base["mask_iou"] + 0.02 or anchor["mask_iou"] >= 0.95,
        "base_non_clothing_rgb_within_15pct": anchor["non_clothing_rgb_preservation"] <= 1.15 * base["non_clothing_rgb_preservation"],
        "within_gaussian_rgb_1_20x": anchor["clothing_rgb_l1"] <= 1.20 * gaussian["clothing_rgb_l1"],
        "within_gaussian_lpips_1_20x": anchor["clothing_lpips"] <= 1.20 * gaussian["clothing_lpips"],
        "within_gaussian_mask_iou_0_03": anchor["mask_iou"] >= gaussian["mask_iou"] - 0.03,
    }
    anchor_ok = all(anchor_checks.values())
    if gaussian_ok and anchor_ok:
        decision = "continue_current_representation"
    elif gaussian_ok:
        decision = "adjust_anchor_or_interpolation_without_garment_layer"
    else:
        decision = "design_garment_gaussian_layer"
    return {
        "gaussian": {"status": "PASS" if gaussian_ok else "FAIL", "checks": gaussian_checks},
        "anchor": {"status": "PASS" if anchor_ok else "FAIL", "checks": anchor_checks},
        "architecture_decision": decision,
        "visual_anomaly_review_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = evaluate_capacity(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
