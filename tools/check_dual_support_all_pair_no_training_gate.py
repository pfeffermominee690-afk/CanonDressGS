"""Real-renderer no-training gate for the dual-support all-pair evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper import run_continuous_control_artifact_root_cause as provenance_tools
from tools.paper import run_dual_support_all_pair_evaluation as runner
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--audit-attempt", type=Path, required=True)
    args = parser.parse_args()
    asset_root, attempt = args.asset_root.resolve(), args.audit_attempt.resolve()
    if attempt.exists():
        raise FileExistsError(f"append-only gate output exists: {attempt}")
    runner.validate_source()
    (attempt / "audits").mkdir(parents=True)
    manifest = runner.read_json(runner.micro.FROZEN_MANIFEST)
    assets_before = verify_manifest(manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    trees_before = runner.frozen_trees(asset_root)
    archives_before = runner.archive_fingerprints()
    with provenance_tools.NoTrainingProvenance(attempt, "all_pair_dual_support_renderer_gate") as provenance:
        runtime_value = provenance_tools.runtime(attempt, asset_root, runner.protocol())
        provenance.capture_frozen_runtime_baseline()
        endpoints = runner.micro.endpoint_residuals(runtime_value)
        hard = runner.micro.hard_geometry_residual(endpoints["O01"], endpoints["O02"], endpoints["O01"], 0.5)
        with torch.inference_mode():
            hard_rgb, hard_alpha, hard_diag = runner.micro.render_branches(
                runtime_value, "O01", "cond_000000", (("O01", hard, 1.0),)
            )
            dual_rgb, dual_alpha, dual_diag = runner.micro.render_branches(
                runtime_value, "O01", "cond_000000",
                (("O01", endpoints["O01"], 0.5), ("O02", endpoints["O02"], 0.5)),
            )
    provenance_record = provenance_tools.aggregate_optimizer_provenance(attempt)
    assets_after = verify_manifest(manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    trees_after = runner.frozen_trees(asset_root)
    archives_after = runner.archive_fingerprints()
    checks = {
        "hard_rgb_finite": bool(torch.isfinite(hard_rgb).all()),
        "hard_alpha_finite": bool(torch.isfinite(hard_alpha).all()),
        "dual_rgb_finite": bool(torch.isfinite(dual_rgb).all()),
        "dual_alpha_finite": bool(torch.isfinite(dual_alpha).all()),
        "dual_gaussian_count_doubled": dual_diag["total_gaussian_count"] == 2 * hard_diag["total_gaussian_count"],
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
        "frozen_assets_unchanged": assets_before == assets_after,
        "frozen_trees_unchanged": trees_before == trees_after,
        "archives_unchanged": archives_before == archives_after,
        "paper_final_zero": provenance_record["paper_final"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ALL-PAIR-NO-TRAINING-GATE-VIOLATION: {checks}")
    report = {
        "schema_version": "canondressgs.research.dual_support_all_pair_no_training_gate.v1",
        "status": "PASS", "task_id": runner.TASK_ID, "test_count": len(checks),
        "tests": [{"name": name, "status": "PASS"} for name in checks],
        "hard_render": {"rgb_shape": list(hard_rgb.shape), "alpha_shape": list(hard_alpha.shape), "diagnostics": hard_diag},
        "dual_render": {"rgb_shape": list(dual_rgb.shape), "alpha_shape": list(dual_alpha.shape), "diagnostics": dual_diag},
        "optimizer_provenance": provenance_record,
        "frozen_assets_before": assets_before, "frozen_assets_after": assets_after,
        "frozen_trees_before": trees_before, "frozen_trees_after": trees_after,
        "archives_before": archives_before, "archives_after": archives_after,
        "paper_final": False,
    }
    runner.micro.atomic_json(attempt / "audits/no_training_gate_test_report.json", report)
    print(json.dumps({
        "status": "PASS", "test_count": len(checks),
        "hard_shape": list(hard_rgb.shape), "dual_shape": list(dual_rgb.shape),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
