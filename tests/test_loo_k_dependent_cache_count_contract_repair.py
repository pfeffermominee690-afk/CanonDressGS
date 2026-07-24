from __future__ import annotations

import copy
import json
import math
import py_compile
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from tools.paper import loo_cache_key_plan
from tools.paper import repair_loo_k_dependent_cache_count_contract as repair
from tools.paper import run_loo_basis_adaptation_experiment as runner


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"


def load(name: str) -> dict:
    return json.loads(
        (RISK / name).read_text(encoding="utf-8"),
        object_pairs_hook=repair.strict_object,
    )


ROOT_CAUSE = load("loo_cache_count_root_cause.json")
TRACKS = load("loo_static_render_track_cache_audit.json")
REPLAY = load("loo_hard_lookup_k_replay.json")
PLAN = load("loo_cache_key_plan_v2.json")
COUNTS = load("loo_expected_counts_cache_repaired.json")
OLD_COUNTS = load("loo_expected_counts_amended.json")
STORAGE = load("loo_storage_forecast_cache_repaired.json")
EXECUTION = load("loo_execution_contract_cache_repaired.json")
TESTS = load("loo_cache_count_repair_tests.json")
SUMMARY = load("loo_cache_count_repair_final_summary.json")
HANDOFF = json.loads(
    (ROOT / "project_control_handoff/loo_cache_count_repair_handoff.json").read_text(
        encoding="utf-8"
    ),
    object_pairs_hook=repair.strict_object,
)
MANIFEST = load("loo_few_view_manifests_repaired.json")
BASIS = load("loo_basis_manifests.json")
OPTIMIZER = load("loo_optimizer_contract.json")
SOURCE = ROOT_CAUSE["source_diagnostic_audit"]
PLANNED = COUNTS["planned_future_counts"]
TASKS = MANIFEST["primary_tasks"]
EXPECTED_MISMATCHES = {
    ("O01", "R0"): ("O04", "O02"),
    ("O01", "R1"): ("O04", "O02"),
    ("O02", "R2"): ("O08", "O01"),
    ("O04", "R0"): ("O03", "O01"),
    ("O08", "R3"): ("O01", "O02"),
}


def test_01_exact_source_branch_and_head() -> None:
    assert SOURCE["source_branch"] == repair.SOURCE_BRANCH
    assert SOURCE["source_diagnostic_head"] == repair.SOURCE_DIAGNOSTIC_HEAD
    assert SOURCE["checks"]["source_branch_head_exact"]


def test_02_historical_execution_result_reporting_heads_absent() -> None:
    assert SOURCE["preserved_head_state"] == {
        "execution_head_frozen": False,
        "final_reporting_head_exists": False,
        "result_head_exists": False,
    }


def test_03_attempt_001_absent() -> None:
    assert SOURCE["checks"]["attempt_001_absent"]
    assert not SUMMARY["attempt_001_exists"]


def test_04_attempt_002_absent() -> None:
    assert SOURCE["checks"]["attempt_002_absent"]
    assert not SUMMARY["attempt_002_exists"]


def test_05_historical_invalid_diagnosis_preserved() -> None:
    assert SOURCE["preserved_classification"] == "LOO_ADAPTATION_EXECUTION_INVALID"
    assert ROOT_CAUSE["previous_classification"] == "LOO_ADAPTATION_EXECUTION_INVALID"


def test_06_old_fourteen_artifacts_immutable() -> None:
    audit = SOURCE["historical_artifact_immutability"]
    assert audit["artifact_count"] == 14
    assert audit["mutation_count"] == 0
    assert all(row["match"] for row in audit["rows"])


def test_07_primary_k_exactly_one_two() -> None:
    assert sorted({int(task["K"]) for task in TASKS}) == [1, 2]


def test_08_k4_unauthorized() -> None:
    assert not MANIFEST["k4_deferred_contract"]["execution_authorized"]
    assert MANIFEST["k4_deferred_contract"]["current_task_count"] == 0


def test_09_five_splits_unchanged() -> None:
    assert [row["held_out_garment"] for row in BASIS["splits"]] == list(runner.OUTFITS)


def test_10_forty_tasks_unchanged() -> None:
    assert len(TASKS) == len({task["task_id"] for task in TASKS}) == 40


def test_11_k1_mappings_unchanged() -> None:
    expected = {
        "R0": "cond_000000", "R1": "cond_000017",
        "R2": "cond_000347", "R3": "cond_000000",
    }
    assert all(
        task["selected_adaptation_conditions"] == [expected[task["rotation"]]]
        for task in TASKS if int(task["K"]) == 1
    )


def test_12_k2_mappings_unchanged() -> None:
    rows = [task for task in TASKS if int(task["K"]) == 2]
    assert len(rows) == 20
    assert all(
        len(task["selected_adaptation_conditions"]) == 2
        and task["selected_adaptation_conditions"] == task["optimize_pool"]
        for task in rows
    )


def test_13_k1_strict_subset_k2_twenty_of_twenty() -> None:
    indexed = {
        (task["held_out_garment"], task["rotation"], int(task["K"])): task
        for task in TASKS
    }
    pairs = [
        (
            indexed[(outfit, rotation, 1)],
            indexed[(outfit, rotation, 2)],
        )
        for outfit in runner.OUTFITS
        for rotation in ("R0", "R1", "R2", "R3")
    ]
    assert len(pairs) == 20
    assert all(
        set(one["selected_adaptation_conditions"])
        < set(two["selected_adaptation_conditions"])
        for one, two in pairs
    )


def test_14_split_overlap_zero() -> None:
    for task in TASKS:
        adaptation = set(task["selected_adaptation_conditions"])
        calibration = set(task["calibration_conditions"])
        test = set(task["test_conditions"])
        assert adaptation.isdisjoint(calibration)
        assert adaptation.isdisjoint(test)
        assert calibration.isdisjoint(test)


def test_15_held_out_boundary_unchanged() -> None:
    assert all(
        task["held_out_garment"] not in task["basis_garments"]
        and all(
            value == 0
            for key, value in task["held_out_information_use"].items()
            if key != "teacher_offline_oracle_only"
        )
        for task in TASKS
    )


def test_16_train_only_f2_inputs() -> None:
    assert REPLAY["classification"] == "DEPLOYABLE_INPUT_DETERMINISTIC_PREFLIGHT"
    assert REPLAY["test_target_reads"] == 0
    assert all(row["held_out_garment"] not in row["centroid_bank"] for row in REPLAY["rows"])


def test_17_four_garment_centroid_banks_only() -> None:
    assert all(len(row["centroid_bank"]) == 4 for row in REPLAY["rows"])


def test_18_k1_feature_aggregation_unchanged() -> None:
    rows = [row for row in REPLAY["rows"] if int(row["K"]) == 1]
    assert len(rows) == 20
    assert all(len(row["adaptation_conditions"]) == 1 for row in rows)


def test_19_k2_feature_aggregation_unchanged() -> None:
    rows = [row for row in REPLAY["rows"] if int(row["K"]) == 2]
    assert len(rows) == 20
    assert all(len(row["adaptation_conditions"]) == 2 for row in rows)


def test_20_squared_l2_distance_unchanged() -> None:
    for row in REPLAY["rows"]:
        distances = row["squared_distances"]
        assert all(math.isfinite(value) and value >= 0 for value in distances.values())
        assert row["selected_known_endpoint"] == min(
            distances, key=lambda name: (distances[name], name)
        )


def test_21_lexical_tie_break_unchanged() -> None:
    assert all(
        row["tie_break"] == "minimum standardized squared L2 then lexical outfit id"
        for row in REPLAY["rows"]
    )


def test_22_fifteen_same_endpoints() -> None:
    assert REPLAY["same_endpoint_pairs"] == 15


def test_23_five_different_endpoints() -> None:
    assert REPLAY["different_endpoint_pairs"] == 5


def test_24_exact_mismatch_identities() -> None:
    actual = {
        (row["held_out_garment"], row["rotation"]): (
            row["K1_selected_known_endpoint"], row["K2_selected_known_endpoint"]
        )
        for row in REPLAY["mismatch_records"]
    }
    assert actual == EXPECTED_MISMATCHES


def test_25_replay_semantic_sha_is_self_consistent() -> None:
    assert repair.canonical_sha(repair._semantic_replay_payload(REPLAY)) == REPLAY[
        "deterministic_replay_sha256"
    ]


def test_26_held_out_teacher_reads_zero() -> None:
    assert REPLAY["held_out_teacher_reads"] == 0
    assert all(row["held_out_teacher_reads"] == 0 for row in REPLAY["rows"])


def test_27_test_metric_and_optimizer_reads_zero() -> None:
    assert REPLAY["test_metric_reads"] == 0
    assert REPLAY["optimizer_reads"] == 0


def test_28_six_static_tracks_enumerated() -> None:
    assert TRACKS["track_count"] == len(TRACKS["rows"]) == 6
    assert {row["track_id"] for row in TRACKS["rows"]} == set(runner.STATIC_METHODS)


def test_29_five_tracks_guaranteed_invariant() -> None:
    assert sum(row["K_invariant_mathematically_guaranteed"] for row in TRACKS["rows"]) == 5


def test_30_one_track_k_dependent() -> None:
    dependent = [
        row for row in TRACKS["rows"]
        if not row["K_invariant_mathematically_guaranteed"]
    ]
    assert [row["track_id"] for row in dependent] == [loo_cache_key_plan.HARD_LOOKUP]


def test_31_logical_requests_54960() -> None:
    assert PLAN["logical_request_count"] == 54_960


def test_32_unique_physical_keys_54845() -> None:
    assert PLAN["unique_physical_key_count"] == 54_845


def test_33_cache_hits_115() -> None:
    assert PLAN["cache_hit_count"] == 115


def test_34_logical_minus_unique_equals_hits() -> None:
    assert PLAN["logical_request_count"] - PLAN["unique_physical_key_count"] == PLAN["cache_hit_count"]


def test_35_invariant_hits_100() -> None:
    assert PLAN["guaranteed_k_invariant_hits"] == 100


def test_36_hard_lookup_hits_15() -> None:
    assert PLAN["hard_lookup_k_shared_hits"] == 15


def test_37_hard_lookup_divergent_pairs_5() -> None:
    assert PLAN["hard_lookup_k_divergent_pairs"] == 5


def test_38_duplicate_keys_are_legal_exact_identity_pairs() -> None:
    assert len(PLAN["duplicate_key_rows"]) == 115
    assert PLAN["key_multiplicity_histogram"]["2"] == 115
    assert all(
        row["source_logical_request_id"] != row["duplicate_logical_request_id"]
        for row in PLAN["duplicate_key_rows"]
    )


def test_39_different_endpoint_pairs_have_different_keys() -> None:
    rows = [
        row for row in PLAN["hard_lookup_selected_endpoint_pairs"]
        if row["K1_selected_known_endpoint"] != row["K2_selected_known_endpoint"]
    ]
    assert len(rows) == 5
    assert all(row["K1_cache_key_v2"] != row["K2_cache_key_v2"] for row in rows)


def test_40_same_endpoint_pairs_have_same_keys() -> None:
    rows = [
        row for row in PLAN["hard_lookup_selected_endpoint_pairs"]
        if row["K1_selected_known_endpoint"] == row["K2_selected_known_endpoint"]
    ]
    assert len(rows) == 15
    assert all(row["K1_cache_key_v2"] == row["K2_cache_key_v2"] for row in rows)


def test_41_plan_deterministic_sha_is_self_consistent() -> None:
    payload = copy.deepcopy(PLAN)
    expected = payload.pop("deterministic_plan_sha256")
    assert loo_cache_key_plan.canonical_sha256(payload) == expected


def test_42_second_independent_replay_identical() -> None:
    evidence = REPLAY["independent_source_replays"]
    assert evidence["count"] == 2
    assert evidence["sha_match"] and evidence["semantic_payload_match"]
    assert len(set(evidence["deterministic_replay_sha256"])) == 1


def test_43_optimizer_runs_120() -> None:
    assert PLANNED["total_optimizer_runs"] == 120


def test_44_optimizer_steps_36000() -> None:
    assert PLANNED["optimizer_steps"] == 36_000


def test_45_forward_backward_36000() -> None:
    assert PLANNED["forward_loss_calls"] == PLANNED["backward_calls"] == 36_000


def test_46_checkpoints_720() -> None:
    assert PLANNED["checkpoint_writes"] == 720


def test_47_evaluations_960() -> None:
    assert PLANNED["evaluation_inference"] == 960


def test_48_table_rows_65() -> None:
    assert PLANNED["wide_result_table_rows"] == 65


def test_49_visual_sheets_26() -> None:
    assert PLANNED["visual_sheets"] == 26


def test_50_only_render_counts_changed() -> None:
    old = OLD_COUNTS["planned_future_counts"]
    common_changes = {
        key for key in old
        if old[key] != PLANNED[key]
    }
    assert common_changes == {"K_shared_static_cache_hits", "unique_physical_renders"}
    assert PLANNED["guaranteed_k_invariant_hits"] == 100
    assert PLANNED["hard_lookup_k_shared_hits"] == 15
    assert PLANNED["hard_lookup_k_divergent_pairs"] == 5


def test_51_storage_uses_54845_physical_renders() -> None:
    assert STORAGE["physical_render_count"] == 54_845


def test_52_storage_delta_matches_five_prediction_pairs() -> None:
    delta = STORAGE["delta_from_five_physical_renders"]
    assert delta["prediction_render_pairs"] == 5
    assert delta["raw_estimated_bytes"] == 5 * 2 * (300 << 10)
    assert delta["required_free_bytes"] == 3_993_600


def test_53_storage_margin_pass() -> None:
    forecast = STORAGE["new_forecast"]
    assert STORAGE["status"] == forecast["status"] == "PASS"
    assert forecast["actual_free_bytes"] >= forecast["required_free_bytes"]


def _semantic_row(suffix: str) -> dict:
    audit = repair.semantic_immutability_audit()
    return next(
        row for row in audit["scientific_artifact_rows"]
        if row["path"].endswith(suffix)
    )


def test_54_basis_protocol_unchanged() -> None:
    assert _semantic_row("loo_basis_manifests.json")["match"]


def test_55_loss_unchanged() -> None:
    assert _semantic_row("loo_adaptation_loss_contract.json")["match"]


def test_56_optimizer_unchanged() -> None:
    assert _semantic_row("loo_optimizer_contract.json")["match"]
    assert OPTIMIZER["shared_budget"]["steps"] == 300


def test_57_full_residual_schema_unchanged() -> None:
    full = OPTIMIZER["methods"]["FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION"]
    assert full["trainable_scalar_count"] == 4_400_000
    assert full["gaussian_count"] == 200_000


def test_58_evaluator_unchanged() -> None:
    assert _semantic_row("loo_evaluator_contract_amended.json")["match"]


def test_59_success_gates_unchanged() -> None:
    assert _semantic_row("loo_success_gates_amended.json")["match"]


def test_60_hard_lookup_ast_semantics_unchanged() -> None:
    audit = repair.semantic_immutability_audit()
    assert audit["hard_lookup_semantic_drift"] == 0
    assert all(row["match"] for row in audit["hard_lookup_semantic_ast_rows"])


def test_61_scientific_semantic_drift_zero() -> None:
    audit = repair.semantic_immutability_audit()
    assert audit["scientific_semantic_drift"] == 0


def test_62_no_scientific_attempt_created() -> None:
    assert SUMMARY["actual_execution_counts"]["scientific_attempts_created"] == 0


def test_63_no_renderer_calls() -> None:
    assert SUMMARY["actual_execution_counts"]["renderer_calls"] == 0
    assert SUMMARY["actual_execution_counts"]["scientific_renders"] == 0


def test_64_no_optimizer_created_or_stepped() -> None:
    assert SUMMARY["actual_execution_counts"]["optimizer_creations"] == 0
    assert SUMMARY["actual_execution_counts"]["optimizer_steps"] == 0


def test_65_no_checkpoints() -> None:
    assert SUMMARY["actual_execution_counts"]["checkpoint_writes"] == 0


def test_66_no_metrics_or_evaluations() -> None:
    assert SUMMARY["actual_execution_counts"]["LOO_metrics"] == 0
    assert SUMMARY["actual_execution_counts"]["evaluations"] == 0


def test_67_no_figure_bank_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Figure_Bank"] == 0


def test_68_no_pure_endpoint_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Pure_Endpoint"] == 0


def test_69_no_headroom_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Coefficient_Headroom"] == 0


def test_70_no_subject00_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Subject00"] == 0


def test_71_credential_findings_zero() -> None:
    assert SUMMARY["credential_findings"] == 0
    paths = [DOCS / name for name in repair.REPORT_NAMES]
    paths += [RISK / name for name in repair.RISK_NAMES]
    paths.append(ROOT / "project_control_handoff/loo_cache_count_repair_handoff.json")
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    patterns = (
        r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9_-]{20,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    assert not any(re.search(pattern, text) for pattern in patterns)


def test_72_json_strict_parse() -> None:
    paths = [RISK / name for name in repair.RISK_NAMES]
    paths.append(ROOT / "project_control_handoff/loo_cache_count_repair_handoff.json")
    assert all(isinstance(repair.read_json(path), dict) for path in paths)


def test_73_yaml_safe_load() -> None:
    value = yaml.safe_load(
        (RISK / "loo_basis_adaptation_protocol_amended.yaml").read_text(encoding="utf-8")
    )
    assert value["primary_budgets"] == [1, 2]


def test_74_markdown_nonempty() -> None:
    assert all(
        len((DOCS / name).read_text(encoding="utf-8").strip()) > 500
        for name in repair.REPORT_NAMES
    )


def test_75_py_compile() -> None:
    paths = (
        ROOT / "tools/paper/loo_cache_key_plan.py",
        ROOT / "tools/paper/repair_loo_k_dependent_cache_count_contract.py",
        ROOT / "tools/paper/run_loo_basis_adaptation_experiment.py",
        Path(__file__),
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, path in enumerate(paths):
            py_compile.compile(
                str(path), cfile=str(Path(directory) / f"module_{index}.pyc"),
                doraise=True,
            )


def test_76_windows_unit_test_evidence_slot() -> None:
    assert TESTS["local_python_311"] in {"PENDING", "PASS"}
    assert TESTS["local_python_310_torch"] in {"PENDING", "PASS"}


def test_77_cloud_unit_test_evidence_slot() -> None:
    assert TESTS["cloud_python_310_torch"] in {"PENDING", "PASS"}


def test_78_deterministic_artifact_regeneration() -> None:
    regenerated = repair.build_plan(REPLAY, COUNTS)
    assert regenerated == PLAN
    assert repair.static_track_audit(REPLAY, regenerated) == TRACKS


def test_79_git_diff_check() -> None:
    committed = subprocess.run(
        ["git", "diff", "--check", repair.SOURCE_DIAGNOSTIC_HEAD, "HEAD", "--"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    working = subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    assert committed.returncode == 0, committed.stdout + committed.stderr
    assert working.returncode == 0, working.stdout + working.stderr


def test_80_local_origin_cloud_consistency_contract() -> None:
    assert SUMMARY["repair_branch"] == repair.REPAIR_BRANCH
    assert HANDOFF["repair_branch"] == repair.REPAIR_BRANCH
    assert HANDOFF["windows_worktree"].endswith(
        "canondressgs_loo_k_dependent_cache_count_contract_repair"
    )
    assert HANDOFF["cloud_worktree"].endswith(
        "canondressgs_loo_k_dependent_cache_count_contract_repair"
    )
    assert SUMMARY["repository_consistency"] in {"PENDING_FINAL_SYNC", "PASS"}


def test_81_both_worktrees_clean_contract() -> None:
    assert HANDOFF["both_worktrees_clean"] in {"PENDING_FINAL_SYNC", True}
    assert not HANDOFF["output_attempt_created"]
    assert SUMMARY["paper_final_count"] == 0
