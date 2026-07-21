from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANUAL = json.loads((ROOT / "paper_protocol/manual_review/all_seed_visual_adjudication.json").read_text(encoding="utf-8"))
STATS = json.loads((ROOT / "paper_protocol/reviewer_risk/ours_vs_a6_paired_statistics.json").read_text(encoding="utf-8"))
REVISION = json.loads((ROOT / "paper_protocol/reviewer_risk/candidate_method_revision.json").read_text(encoding="utf-8"))
B5 = json.loads((ROOT / "paper_protocol/reviewer_risk/b5_contract_audit.json").read_text(encoding="utf-8"))
VIEW = json.loads((ROOT / "paper_protocol/reviewer_risk/view_isolation_audit.json").read_text(encoding="utf-8"))
RISK_REGISTRY = yaml.safe_load((ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml").read_text(encoding="utf-8"))


def _risk_experiments(prefix: str):
    return [item for item in RISK_REGISTRY["experiments"] if item["experiment_id"].startswith(prefix)]


def test_all_formal_seed_visuals_are_reviewed():
    rows = MANUAL["contact_index"]
    assert len(rows) == 51 and len({row["experiment_id"] for row in rows}) == 51
    assert all(row["reviewed"] and len(row["sha256"]) == 64 for row in rows)
    assert MANUAL["summary"]["formal_contact_sheets_reviewed"] == 51
    assert len(MANUAL["ours_a6_swap_index"]) == 6
    assert all(row["reviewed"] for row in MANUAL["ours_a6_swap_index"])


def test_ours_a6_uses_paired_episode_metrics():
    contract = STATS["source_contract"]
    assert contract["paired_episode_count"] == 60 and contract["selective_exclusion"] is False
    assert len(contract["raw_sources"]["ours"]) == len(contract["raw_sources"]["a6"]) == 3
    assert all(item["episode_count"] == 20 for family in contract["raw_sources"].values() for item in family)
    assert all(metric["n_paired_episodes"] == 60 for metric in STATS["metrics"].values())


def test_a6_differs_only_by_pairwise_geometry():
    pairs = STATS["source_contract"]["method_config_snapshots"]
    assert len(pairs) == 3 and all(item["byte_identical"] for item in pairs)
    text = STATS["source_contract"]["execution_difference"]
    assert "0.10 for Ours" in text and "0.0 for A6" in text and "otherwise the same" in text
    runtime = (ROOT / "tools/paper/formal_batch_runtime.py").read_text(encoding="utf-8")
    assert 'weight = 0.0 if method.startswith("A6_") else 0.10' in runtime


def test_pairwise_geometry_is_rejected_and_ours_v2_is_proposed():
    assert STATS["decision"] == REVISION["pairwise_geometry_decision"] == "PAIRWISE_GEOMETRY_REJECTED"
    assert REVISION["ours_v2"]["status"] == "PROPOSED_NOT_RUN"
    assert REVISION["ours_v2"]["pairwise_geometry_weight"] == 0.0
    assert REVISION["old_ours"]["evidence_status"] == "HISTORICAL_EVIDENCE"


def test_b1_b2_identity_is_reported():
    audit = STATS["b1_b2_audit"]
    assert audit["aggregate_table_values_equal_at_reported_precision"] is True
    assert audit["bitwise_identical"] is False
    assert audit["numerically_completely_identical"] is False
    assert "Optimization Upper Bound" in audit["b1_contract"]
    assert "not reference inference" in audit["b2_contract"]


def test_b6_uses_reference_not_outfit_id():
    rows = _risk_experiments("RR-B6-")
    assert len(rows) == 3 and {row["seed"] for row in rows} == {0, 1, 2}
    assert all(row["model_input"] == "frozen_F2_reference_mean_max_only" for row in rows)
    assert all(row["outfit_id_input"] is False for row in rows)
    assert all(row["correct_episode_count"] == 20 and row["swap_episode_count"] == 80 for row in rows)


def test_b7_has_no_outfit_id():
    row = _risk_experiments("RR-B7-")[0]
    assert row["outfit_id_input"] is False and row["optimizer_steps"] == 0
    assert row["model_input"] == "frozen_F2_reference_mean_max_only"


def test_complex_corrected_matrix_is_complete():
    assert set(B5["fair_2x2_matrix"]) == {
        "M1_linear_corrected", "M2_linear_legacy_endpoint",
        "M3_complex_corrected", "M4_complex_legacy_endpoint",
    }
    assert B5["fair_2x2_matrix"]["M3_complex_corrected"]["status"] == "MISSING_NOT_RUN"
    assert B5["fair_2x2_matrix"]["M4_complex_legacy_endpoint"]["status"] == "MISSING_NOT_RUN"
    assert len(_risk_experiments("RR-M3-")) == len(_risk_experiments("RR-M4-")) == 3


def test_b5_contract_is_not_ambiguous_or_is_flagged():
    assert B5["classification"] in {"B5_CONTRACT_RESOLVED", "B5_CONTRACT_AMBIGUOUS"}
    assert B5["actual_b5_contract"]["actual_matrix_cell"] == "OFF_MATRIX_COMPLEX_PLUS_PAIRWISE_GEOMETRY"
    assert B5["claim_boundary"]["may_claim_complex_fusion_is_worse_from_b5_alone"] is False


def test_color_counterfactuals_do_not_change_target():
    row = _risk_experiments("RR-COLOR-")[0]
    assert row["optimizer_steps"] == 0 and row["target_pose_camera_changed"] is False
    assert len(row["transformations"]) == 7 and row["transformations"][0] == "C0_original"
    protocol = (ROOT / "docs/PAPER/AAAI27_FAIR_BASELINE_AND_COUNTERFACTUAL_PROTOCOL_20260721.md").read_text(encoding="utf-8")
    assert "变换只作用于 reference RGB" in protocol


def test_extended_metrics_are_evaluation_only():
    row = _risk_experiments("RR-EXTENDED-")[0]
    assert row["optimizer_steps"] == 0 and row["existing_renders_only"] is True
    assert set(row["methods"]) == {"Ours-v2", "B1", "B2", "B4", "B6", "B7"}


def test_view_isolation_classification_is_explicit():
    assert VIEW["classification"] == "VIEW-TRANSDUCTIVE"
    assert VIEW["classification_is_explicit_and_unique"] is True
    assert VIEW["reasoning"]["direct_prediction_forward_leakage"] is False


def test_strict_view_experiment_is_blocked_by_default():
    canary = VIEW["strict_view_one_fold_canary"]
    assert canary["status"] == "BLOCKED_PENDING_AUTHORIZATION" and canary["auto_execute"] is False
    registry_canary = _risk_experiments("RR-STRICT-")[0]
    assert registry_canary["status"] == "BLOCKED_PENDING_AUTHORIZATION"


def test_reviewer_registry_starts_not_run():
    rows = RISK_REGISTRY["experiments"]
    assert len(rows) == 16
    assert all(item["status"] == "NOT_RUN" for item in rows if not item["experiment_id"].startswith("RR-STRICT-"))
    assert RISK_REGISTRY["auto_execute"] is False and RISK_REGISTRY["paper_final_transition_allowed"] is False


def test_no_optimizer_is_created():
    source = (ROOT / "tools/paper/build_reviewer_risk_evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    assert "torch" not in imported and "torch.optim" not in source
    assert REVISION["execution_guards"]["optimizer_created_in_this_task"] is False


def test_cuda_is_not_initialized():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert REVISION["execution_guards"]["cuda_initialized_in_this_task"] is False
    assert VIEW["execution_guards"]["cuda_initialized_in_this_task"] is False


def test_formal_registry_is_unchanged():
    path = ROOT / "paper_protocol/experiment_registry.yaml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == MANUAL["immutability"]["formal_registry_sha256_before"]
    assert digest == MANUAL["immutability"]["formal_registry_sha256_after"]
    manifest = hashlib.sha256((ROOT / "paper_protocol/frozen_asset_manifest.json").read_bytes()).hexdigest()
    assert manifest == MANUAL["immutability"]["frozen_manifest_sha256_before"]
    assert manifest == MANUAL["immutability"]["frozen_manifest_sha256_after"]


def test_formal_outputs_are_immutable():
    immutable = MANUAL["immutability"]
    assert immutable["formal_output_metadata_tree_before"] == immutable["formal_output_metadata_tree_after"]
    assert REVISION["execution_guards"]["formal_outputs_modified"] is False


def test_paper_final_is_never_set():
    assert MANUAL["paper_final"] is False
    assert REVISION["old_ours"]["paper_final"] is False
    assert RISK_REGISTRY["paper_final_transition_allowed"] is False


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} reviewer-risk adjudication tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
