from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sprint.build_o01_o08_two_outfit_manifest import sha256_file  # noqa: E402
from tools.sprint.build_two_outfit_discrimination_contact_sheet import (  # noqa: E402
    build_contact_sheet,
)
from tools.sprint.evaluate_reference_swap_discrimination import (  # noqa: E402
    build_reference_swap_plan,
    score_reference_swap_plan,
)
from tools.sprint.measure_two_outfit_conditioning_collapse import (  # noqa: E402
    _load_tensor_bundle,
    measure_conditioning_collapse,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified CPU-only entrypoint for prepared two-outfit evaluation artifacts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-bundle", type=Path)
    parser.add_argument("--target-bundle", type=Path)
    parser.add_argument("--collapse-first", type=Path)
    parser.add_argument("--collapse-second", type=Path)
    parser.add_argument("--residual-bounds-json", type=Path)
    parser.add_argument("--garment-render-mask", type=Path)
    parser.add_argument("--contact-layout", type=Path)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = build_reference_swap_plan(manifest)
    plan_path = output / "reference_swap_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "reference_swap_plan": str(plan_path),
        "case_counts": plan["case_counts"],
        "training_started": False,
        "optimizer_created": False,
        "gpu_used": False,
    }

    if (args.prediction_bundle is None) != (args.target_bundle is None):
        raise ValueError("prediction and target bundles must be provided together")
    if args.prediction_bundle is not None:
        predictions = torch.load(args.prediction_bundle, map_location="cpu", weights_only=False)
        targets = torch.load(args.target_bundle, map_location="cpu", weights_only=False)
        metrics = score_reference_swap_plan(plan, predictions, targets)
        metrics_path = output / "reference_swap_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["reference_swap_metrics"] = str(metrics_path)

    collapse_arguments = (args.collapse_first, args.collapse_second, args.residual_bounds_json)
    if any(value is not None for value in collapse_arguments) and not all(value is not None for value in collapse_arguments):
        raise ValueError("collapse-first, collapse-second, and residual-bounds-json are an atomic group")
    if args.collapse_first is not None:
        bounds = json.loads(args.residual_bounds_json.read_text(encoding="utf-8"))
        render_mask = None
        if args.garment_render_mask is not None:
            render_mask = torch.load(args.garment_render_mask, map_location="cpu", weights_only=False)
        collapse = measure_conditioning_collapse(
            _load_tensor_bundle(args.collapse_first),
            _load_tensor_bundle(args.collapse_second),
            residual_bounds=bounds,
            garment_render_mask=render_mask,
            thresholds=None,
        )
        collapse_path = output / "conditioning_collapse_metrics.json"
        collapse_path.write_text(json.dumps(collapse, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["conditioning_collapse_metrics"] = str(collapse_path)

    if args.contact_layout is not None:
        summary["contact_sheet"] = build_contact_sheet(args.contact_layout, output / "contact_sheet")

    summary_path = output / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
