from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from tools.paper import p0_candidate_runner as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "42a28b386f6f32e23e5e16680408820efd8a65c7"
MANIFEST_PATH = ROOT / "paper_protocol/reviewer_risk/p0_candidate_runtime_manifest.yaml"
FORWARD_AUDIT = ROOT / "paper_protocol/reviewer_risk/p0_candidate_forward_boundary_audit.json"
OPTIMIZER_AUDIT = ROOT / "paper_protocol/reviewer_risk/p0_candidate_optimizer_provenance.json"
EVALUATOR_AUDIT = ROOT / "paper_protocol/reviewer_risk/p0_candidate_evaluator_compatibility.json"
M4_AUDIT = ROOT / "paper_protocol/reviewer_risk/m4_a5_supervision_contract_parity.json"


def _canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _historical_sha(relative: str) -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "show", f"{SOURCE_HEAD}:{relative}"], cwd=ROOT)
    ).hexdigest()


def test_p0_runtime_manifest_has_13_candidates() -> None:
    manifest = runner.load_runtime_manifest(MANIFEST_PATH)
    assert manifest["candidate_count"] == len(manifest["candidates"]) == 13
    assert sum(row["trainable"] for row in manifest["candidates"]) == 12
    assert sum(not row["trainable"] for row in manifest["candidates"]) == 1


def test_reviewer_registry_is_not_modified() -> None:
    path = ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
    assert _canonical_sha(path) == _historical_sha(
        "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
    ) == runner.REVIEWER_REGISTRY_SHA256


def test_candidate_runner_has_only_allowed_commands() -> None:
    source = (ROOT / "tools/paper/p0_candidate_runner.py").read_text(encoding="utf-8")
    for command in ("validate", "plan", "dry-run", "status"):
        assert f'subparsers.add_parser("{command}")' in source
    for command in ("train", "resume", "evaluate", "aggregate", "export-paper"):
        assert f'subparsers.add_parser("{command}")' not in source


def test_candidate_runner_plan_contains_all_candidates() -> None:
    plan = runner.runtime_plan(runner.load_runtime_manifest(MANIFEST_PATH))
    assert plan["candidate_count"] == 13
    assert plan["planned_total_optimizer_steps"] == 3600
    assert plan["executed_optimizer_steps"] == 0
    assert plan["formal_run_authorized"] is False


def test_candidate_runner_dry_run_has_no_backward() -> None:
    source = (ROOT / "tools/paper/p0_candidate_runner.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.autograd.grad" not in source


def test_candidate_runner_has_no_optimizer_step() -> None:
    source = (ROOT / "tools/paper/p0_candidate_runner.py").read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
    assert "optimizer.zero_grad(" not in source


def test_candidate_and_legacy_optimizers_are_separate() -> None:
    audit = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))
    assert "candidate_optimizer" in audit
    assert "legacy_context_optimizer" in audit
    assert audit["candidate_optimizer"]["step_count"] == 0
    assert audit["legacy_context_optimizer"]["step_count"] == 0


def test_candidate_optimizer_parameter_counts_are_explicit() -> None:
    counts = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))["candidate_optimizer"]["parameter_counts"]
    assert counts == {"Ours-v2": 3076, "B6": 3589, "M3": 234771, "M4": 234771}


def test_candidate_frozen_legacy_overlap_is_zero() -> None:
    overlap = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))["overlap"]
    assert set(overlap.values()) == {0}


def test_target_appearance_is_not_in_prediction_forward_audit() -> None:
    audit = json.loads(FORWARD_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert {"target_rgb", "target_mask"}.issubset(audit["prediction_branch_forbidden_inputs"])
    assert all(row["target_forward_input_used"] is False for row in audit["candidates"].values())


def test_target_pose_camera_are_not_in_coefficient_branch_audit() -> None:
    audit = json.loads(FORWARD_AUDIT.read_text(encoding="utf-8"))
    assert {"target_pose", "target_camera"}.issubset(audit["prediction_branch_forbidden_inputs"])
    assert audit["downstream_boundary"]["target_pose_camera_allowed_after_garment_prediction"] is True
    assert audit["downstream_boundary"]["classified_as_prediction_leakage"] is False


def test_correct_episode_set_is_exact_and_unique() -> None:
    audit = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))
    keys = [(outfit, view) for outfit in audit["seen_outfits"] for view in audit["fixed_views"]]
    assert len(keys) == len(set(keys)) == audit["correct_episode_contract"]["expected_count"] == 20


def test_swap_tuple_set_is_exact_and_unique() -> None:
    audit = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))
    tuples = [
        (target, source, view)
        for target in audit["seen_outfits"]
        for source in audit["seen_outfits"]
        if source != target
        for view in audit["fixed_views"]
    ]
    assert len(tuples) == len(set(tuples)) == audit["swap_tuple_contract"]["expected_count"] == 80


def test_robustness_schema_is_complete() -> None:
    robustness = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))["robustness_contract"]
    assert set(robustness) == {
        "permutation", "single_reference", "two_reference_dropout",
        "zero_replacement", "base_replacement",
    }
    assert all(row["required"] and row["minimum_record_count"] > 0 for row in robustness.values())


def test_aggregation_distinguishes_replicate_and_seed() -> None:
    aggregation = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))["aggregation"]
    assert aggregation["Ours-v2"]["axis"] == "replicate"
    assert aggregation["Ours-v2"]["independent_random_initialization_sample_std_allowed"] is False
    assert all(aggregation[name]["axis"] == "seed" for name in ("B6", "M3", "M4"))
    assert aggregation["B7"]["axis"] == "fixed"


def test_m4_matches_a5_supervision_contract_report() -> None:
    audit = json.loads(M4_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert all(audit["field_parity"].values())
    assert audit["m4"]["only_primary_change"] == "Linear fusion -> Complex fusion"
    assert audit["historical_source"]["sha256"] == _historical_sha("tools/paper/formal_batch_runtime.py")


def test_b5_remains_off_matrix_historical_report() -> None:
    audit = json.loads(M4_AUDIT.read_text(encoding="utf-8"))
    assert audit["historical_b5_role"] == "OFF_MATRIX_HISTORICAL_EVIDENCE"


def test_manifest_uses_only_prefight_statuses() -> None:
    manifest = runner.load_runtime_manifest(MANIFEST_PATH)
    statuses = {row["status"] for row in manifest["candidates"]}
    assert statuses == {"PREFLIGHT_READY"}
    assert not statuses.intersection(manifest["forbidden_statuses"])


def test_formal_registry_is_unchanged() -> None:
    path = ROOT / "paper_protocol/experiment_registry.yaml"
    assert _canonical_sha(path) == _historical_sha("paper_protocol/experiment_registry.yaml") == runner.FORMAL_REGISTRY_SHA256


def test_formal_outputs_are_immutable() -> None:
    audit = json.loads((ROOT / "paper_protocol/reviewer_risk/no_training_gate.json").read_text(encoding="utf-8"))
    assert audit["formal_outputs_unchanged"] is True
    assert audit["after"]["formal_output_metadata_sha256"] == runner.FORMAL_OUTPUT_SHA256
    assert audit["after"]["formal_output_file_count"] == runner.FORMAL_FILE_COUNT
    assert audit["after"]["formal_output_total_bytes"] == runner.FORMAL_TOTAL_BYTES


def test_previous_protocol_outputs_are_immutable() -> None:
    assert runner.PREVIOUS_PROTOCOL_OUTPUT_SHA256 == "642cd8fa42b5879ae6a212941272ea24aec8ed6eb7e80ba5d8d2f74d4cf10960"
    assert runner.PREVIOUS_PROTOCOL_OUTPUT_FILE_COUNT == 12
    assert runner.PREVIOUS_PROTOCOL_OUTPUT_TOTAL_BYTES == 306862


def test_previous_protocol_artifacts_are_immutable() -> None:
    assert runner.deterministic_protocol_artifact_sha256() == runner.DETERMINISTIC_PROTOCOL_ARTIFACT_SHA256


def test_no_paper_final_is_created() -> None:
    formal = yaml.safe_load((ROOT / "paper_protocol/experiment_registry.yaml").read_text(encoding="utf-8"))
    reviewer = yaml.safe_load((ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml").read_text(encoding="utf-8"))
    assert sum(row.get("status") == "PAPER_FINAL" for row in formal["experiments"] + reviewer["experiments"]) == 0
    assert reviewer["paper_final_transition_allowed"] is False


def test_runner_creates_no_formal_artifact_types() -> None:
    source = (ROOT / "tools/paper/p0_candidate_runner.py").read_text(encoding="utf-8")
    assert "torch.save(" not in source
    assert "save_render_tensor" not in source
    assert "evaluate_records(" not in source


def test_evaluator_compatibility_preserves_27_metrics() -> None:
    audit = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))
    assert audit["metric_schema"]["existing_metric_count"] == 27
    assert audit["metric_schema"]["new_metrics_added"] is False
    assert audit["evaluator_executed"] is False


def test_o07_is_excluded_from_seen_macro() -> None:
    held_out = json.loads(EVALUATOR_AUDIT.read_text(encoding="utf-8"))["held_out"]
    assert held_out["O07_in_seen_macro"] is False
    assert held_out["allowed_role"] == "held_out_diagnostic_only"
