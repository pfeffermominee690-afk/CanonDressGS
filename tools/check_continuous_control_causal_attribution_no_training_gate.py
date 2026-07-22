from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.paper import run_continuous_control_artifact_root_cause as previous
from tools.paper import run_continuous_control_causal_attribution as runner
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the real renderer under the causal no-training gate")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--audit-attempt", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    attempt = args.audit_attempt.resolve()
    if attempt.exists():
        raise FileExistsError(f"append-only causal gate audit already exists: {attempt}")
    (attempt / "audits").mkdir(parents=True)
    frozen_manifest = runner.read_json(runner.FROZEN_MANIFEST)
    before_assets = verify_manifest(frozen_manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    before_trees = {
        "formal": previous.tree_manifest(asset_root / sealed.FORMAL_NAME),
        "p0": previous.tree_manifest(asset_root / sealed.P0_NAME),
        "sealed_evaluation": previous.tree_manifest(asset_root / sealed.OUTPUT_NAME),
        "previous_root_cause": previous.tree_manifest(runner.PREVIOUS_OUTPUT),
    }
    with previous.NoTrainingProvenance(attempt, "causal_gate_renderer_smoke") as provenance:
        runtime_value = runner.runtime(attempt, asset_root, runner.protocol())
        provenance.capture_frozen_runtime_baseline()
        residual = runtime_value.basis(runtime_value.coefficients["O01"], chunk_size=16384)
        with torch.inference_mode():
            rgb, alpha = sealed.p0._render(runtime_value.context, "O01", "cond_000000", residual)
        torch.cuda.synchronize()
        renderer_smoke = {
            "rgb_shape": list(rgb.shape),
            "alpha_shape": list(alpha.shape),
            "rgb_finite": bool(torch.isfinite(rgb).all()),
            "alpha_finite": bool(torch.isfinite(alpha).all()),
            "rgb_min": float(rgb.min()),
            "rgb_max": float(rgb.max()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
        }
        if not renderer_smoke["rgb_finite"] or not renderer_smoke["alpha_finite"]:
            raise RuntimeError("renderer inference smoke produced non-finite output")
    provenance_path = attempt / "audits/optimizer_provenance_causal_gate_renderer_smoke.json"
    provenance_record = runner.read_json(provenance_path)
    after_assets = verify_manifest(frozen_manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    after_trees = {
        "formal": previous.tree_manifest(asset_root / sealed.FORMAL_NAME),
        "p0": previous.tree_manifest(asset_root / sealed.P0_NAME),
        "sealed_evaluation": previous.tree_manifest(asset_root / sealed.OUTPUT_NAME),
        "previous_root_cause": previous.tree_manifest(runner.PREVIOUS_OUTPUT),
    }
    checks = {
        "diagnostic_optimizer_created_false": provenance_record["diagnostic_optimizer"]["created"] is False,
        "diagnostic_optimizer_step_zero": provenance_record["diagnostic_optimizer"]["step_count"] == 0,
        "legacy_optimizer_created": provenance_record["legacy_context_optimizer"]["created"] is True,
        "legacy_zero_grad_zero": provenance_record["legacy_context_optimizer"]["zero_grad_count"] == 0,
        "legacy_step_zero": provenance_record["legacy_context_optimizer"]["step_count"] == 0,
        "legacy_scheduler_zero": provenance_record["legacy_context_optimizer"]["scheduler_step_count"] == 0,
        "legacy_state_saved_false": provenance_record["legacy_context_optimizer"]["state_saved"] is False,
        "legacy_discarded": provenance_record["legacy_context_optimizer"]["discarded"] is True,
        "backward_zero": provenance_record["backward_count"] == 0,
        "checkpoint_write_zero": provenance_record["checkpoint_write_count"] == 0,
        "frozen_parameter_change_zero": provenance_record["frozen_parameter_change"] == 0,
        "frozen_assets_unchanged": before_assets == after_assets,
        "frozen_trees_unchanged": before_trees == after_trees,
        "renderer_inference_smoke": renderer_smoke["rgb_finite"] and renderer_smoke["alpha_finite"],
        "paper_final_zero": provenance_record["paper_final"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CAUSAL-ATTRIBUTION-TRUE-NO-TRAINING-GATE-VIOLATION: {checks}")
    report = {
        "schema_version": "canondressgs.research.continuous_control_causal_no_training_gate_tests.v1",
        "status": "PASS",
        "task_id": runner.TASK_ID,
        "test_count": len(checks),
        "tests": [{"name": name, "status": "PASS"} for name in checks],
        "renderer_smoke": renderer_smoke,
        "optimizer_provenance": provenance_record,
        "frozen_assets_before": before_assets,
        "frozen_assets_after": after_assets,
        "frozen_trees_before": before_trees,
        "frozen_trees_after": after_trees,
        "paper_final": False,
    }
    runner.atomic_json(attempt / "audits/no_training_gate_test_report.json", report)
    print(json.dumps({"status": "PASS", "test_count": len(checks), "renderer_smoke": renderer_smoke}, sort_keys=True))


if __name__ == "__main__":
    main()
