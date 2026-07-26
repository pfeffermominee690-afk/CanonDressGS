from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper import run_continuous_control_artifact_root_cause as runner
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


def test_no_diagnostic_optimizer_is_created(record: Mapping[str, Any]) -> None:
    assert record["diagnostic_optimizer"] == {
        "created": False,
        "parameter_count": 0,
        "zero_grad_count": 0,
        "step_count": 0,
        "state_saved": False,
    }


def test_legacy_context_optimizer_is_allowed_and_recorded(record: Mapping[str, Any]) -> None:
    legacy = record["legacy_context_optimizer"]
    assert legacy["created"] is True
    assert legacy["creation_count"] >= 1
    assert all(instance["class"].endswith(".Adam") for instance in legacy["instances"])
    assert all(instance["group_count"] > 0 for instance in legacy["instances"])
    assert all(instance["parameter_count"] > 0 for instance in legacy["instances"])
    assert all(instance["creation_stack"] for instance in legacy["instances"])


def test_legacy_context_optimizer_has_zero_steps(record: Mapping[str, Any]) -> None:
    assert record["legacy_context_optimizer"]["step_count"] == 0


def test_legacy_context_optimizer_has_zero_zero_grad_calls(record: Mapping[str, Any]) -> None:
    assert record["legacy_context_optimizer"]["zero_grad_count"] == 0


def test_legacy_context_optimizer_is_discarded(record: Mapping[str, Any]) -> None:
    assert record["legacy_context_optimizer"]["discarded"] is True
    assert all(instance["lifecycle"] == "CREATED_BY_LEGACY_CONTEXT_HELPER_AND_DISCARDED_BY_CALLER" for instance in record["legacy_context_optimizer"]["instances"])


def test_diagnostic_parameters_do_not_overlap_legacy_optimizer(record: Mapping[str, Any]) -> None:
    assert record["legacy_context_optimizer"]["diagnostic_overlap_count"] == 0


def test_no_backward_occurs(record: Mapping[str, Any]) -> None:
    assert record["backward_count"] == 0


def test_no_scheduler_step_occurs(record: Mapping[str, Any]) -> None:
    assert record["legacy_context_optimizer"]["scheduler_step_count"] == 0


def test_no_checkpoint_is_written(record: Mapping[str, Any]) -> None:
    assert record["checkpoint_write_count"] == 0
    assert record["legacy_context_optimizer"]["state_saved"] is False


def test_teacher_basis_and_frozen_assets_are_unchanged(
    before_assets: Mapping[str, Any],
    after_assets: Mapping[str, Any],
    before_trees: Mapping[str, Any],
    after_trees: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    assert before_assets["status"] == after_assets["status"] == "PASS"
    assert before_assets == after_assets
    assert before_trees == after_trees
    assert record["frozen_parameter_change"] == 0
    assert all(
        instance["frozen_baseline_captured"] is True
        and instance["parameter_fingerprint_before"] == instance["parameter_fingerprint_after"]
        for instance in record["legacy_context_optimizer"]["instances"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the real frozen context under the repaired no-training gate")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--audit-attempt", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    attempt = args.audit_attempt.resolve()
    if attempt.exists():
        raise FileExistsError(f"append-only gate-repair audit already exists: {attempt}")
    (attempt / "audits").mkdir(parents=True)
    frozen_manifest = runner.read_json(runner.FROZEN_MANIFEST)
    before_assets = verify_manifest(frozen_manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    before_trees = {
        "formal": runner.tree_manifest(asset_root / runner.sealed.FORMAL_NAME),
        "p0": runner.tree_manifest(asset_root / runner.sealed.P0_NAME),
        "sealed_evaluation": runner.tree_manifest(asset_root / runner.sealed.OUTPUT_NAME),
    }
    frozen_protocol = runner.protocol()
    with runner.NoTrainingProvenance(attempt, "gate_repair_test") as provenance:
        runtime_value = runner.runtime(attempt, asset_root, frozen_protocol)
        assert runtime_value.device.type == "cuda"
        provenance.capture_frozen_runtime_baseline()
    record = runner.read_json(attempt / "audits/optimizer_provenance_gate_repair_test.json")
    after_assets = verify_manifest(frozen_manifest, runner.PROJECT_ROOT, asset_root, verify_external=True)
    after_trees = {
        "formal": runner.tree_manifest(asset_root / runner.sealed.FORMAL_NAME),
        "p0": runner.tree_manifest(asset_root / runner.sealed.P0_NAME),
        "sealed_evaluation": runner.tree_manifest(asset_root / runner.sealed.OUTPUT_NAME),
    }
    tests = [
        ("test_no_diagnostic_optimizer_is_created", lambda: test_no_diagnostic_optimizer_is_created(record)),
        ("test_legacy_context_optimizer_is_allowed_and_recorded", lambda: test_legacy_context_optimizer_is_allowed_and_recorded(record)),
        ("test_legacy_context_optimizer_has_zero_steps", lambda: test_legacy_context_optimizer_has_zero_steps(record)),
        ("test_legacy_context_optimizer_has_zero_zero_grad_calls", lambda: test_legacy_context_optimizer_has_zero_zero_grad_calls(record)),
        ("test_legacy_context_optimizer_is_discarded", lambda: test_legacy_context_optimizer_is_discarded(record)),
        ("test_diagnostic_parameters_do_not_overlap_legacy_optimizer", lambda: test_diagnostic_parameters_do_not_overlap_legacy_optimizer(record)),
        ("test_no_backward_occurs", lambda: test_no_backward_occurs(record)),
        ("test_no_scheduler_step_occurs", lambda: test_no_scheduler_step_occurs(record)),
        ("test_no_checkpoint_is_written", lambda: test_no_checkpoint_is_written(record)),
        (
            "test_teacher_basis_and_frozen_assets_are_unchanged",
            lambda: test_teacher_basis_and_frozen_assets_are_unchanged(
                before_assets, after_assets, before_trees, after_trees, record
            ),
        ),
    ]
    results = []
    for name, function in tests:
        function()
        results.append({"name": name, "status": "PASS"})
    report = {
        "schema_version": "canondressgs.research.continuous_control_no_training_gate_tests.v1",
        "status": "PASS",
        "task_id": "AAAI27-CONTINUOUS-CONTROL-ROOT-CAUSE-GATE-REPAIR-001",
        "test_count": len(results),
        "tests": results,
        "optimizer_provenance": record,
        "frozen_assets_before": before_assets,
        "frozen_assets_after": after_assets,
        "frozen_trees_before": before_trees,
        "frozen_trees_after": after_trees,
        "paper_final": False,
    }
    runner.atomic_json(attempt / "audits/no_training_gate_test_report.json", report)
    print(json.dumps({"status": "PASS", "test_count": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
