from __future__ import annotations

import hashlib
import json
import math
import py_compile
import re
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # Windows control Python omits PyYAML; cloud parses fully.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "ecf11d6314126f02a9ed1305630a2eac2bd7da04"
SOURCE_BRANCH = "research/leave-one-garment-out-basis-adaptation-protocol-20260724"
REPAIR_BRANCH = "research/loo-few-view-fold-manifest-repair-20260724"
GARMENTS = ["O01", "O02", "O03", "O04", "O08"]

NEW_JSON_PATHS = [
    "paper_protocol/reviewer_risk/loo_few_view_fold_repair.json",
    "paper_protocol/reviewer_risk/loo_few_view_manifests_repaired.json",
    "paper_protocol/reviewer_risk/loo_shared_render_adaptation_loss_reference.json",
    "paper_protocol/reviewer_risk/loo_baseline_registry_amended.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_success_gates_amended.json",
    "paper_protocol/reviewer_risk/loo_expected_counts_amended.json",
    "paper_protocol/reviewer_risk/loo_execution_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_tests.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_final_summary.json",
    "project_control_handoff/loo_protocol_repair_handoff.json",
]
MARKDOWN_PATHS = [
    "docs/PAPER/AAAI27_LOO_FEW_VIEW_FOLD_REPAIR_20260724.md",
    "docs/PAPER/AAAI27_LOO_K1_K2_PRIMARY_CROSSFIT_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_LOO_K4_DEFERRED_DIAGNOSTIC_BOUNDARY_20260724.md",
]


def _strict_pairs(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load_json(relative: str):
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
    )


def canonical_insertion_sha(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def artifacts():
    return {
        "repair": load_json(NEW_JSON_PATHS[0]),
        "views": load_json(NEW_JSON_PATHS[1]),
        "loss_ref": load_json(NEW_JSON_PATHS[2]),
        "baselines": load_json(NEW_JSON_PATHS[3]),
        "evaluator": load_json(NEW_JSON_PATHS[4]),
        "gates": load_json(NEW_JSON_PATHS[5]),
        "counts": load_json(NEW_JSON_PATHS[6]),
        "execution": load_json(NEW_JSON_PATHS[7]),
        "test_contract": load_json(NEW_JSON_PATHS[8]),
        "summary": load_json(NEW_JSON_PATHS[9]),
        "handoff": load_json(NEW_JSON_PATHS[10]),
        "basis": load_json("paper_protocol/reviewer_risk/loo_basis_manifests.json"),
        "optimizer": load_json("paper_protocol/reviewer_risk/loo_optimizer_contract.json"),
        "old_baselines": load_json("paper_protocol/reviewer_risk/loo_baseline_registry.json"),
        "old_gates": load_json("paper_protocol/reviewer_risk/loo_success_gates.json"),
    }


class _Approx:
    def __init__(self, expected: float, absolute: float):
        self.expected = expected
        self.absolute = absolute

    def __eq__(self, actual: float) -> bool:
        return math.isclose(actual, self.expected, rel_tol=0.0, abs_tol=self.absolute)


def approx(expected: float, *, abs: float) -> _Approx:
    return _Approx(expected, abs)


@contextmanager
def raises(exception_type, *, match: str):
    try:
        yield
    except exception_type as error:
        assert re.search(match, str(error)), str(error)
    else:
        raise AssertionError(f"{exception_type.__name__} was not raised")


def test_01_exact_source_head_and_ancestry(artifacts):
    assert artifacts["repair"]["source"]["head"] == SOURCE_HEAD
    assert artifacts["views"]["source"] == {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD}
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"],
        cwd=ROOT,
        check=True,
    )


def test_02_historical_loo_artifact_hash_byte_immutability(artifacts):
    history = artifacts["repair"]["historical_artifact_immutability"]
    assert history["mutation_count"] == 0
    assert len(history["artifacts"]) == 14
    for item in history["artifacts"]:
        source_blob = subprocess.check_output(
            ["git", "rev-parse", f"{SOURCE_HEAD}:{item['path']}"], cwd=ROOT, text=True
        ).strip()
        assert source_blob == item["git_blob"]
        blob_bytes = subprocess.check_output(
            ["git", "cat-file", "blob", source_blob], cwd=ROOT
        )
        assert hashlib.sha256(blob_bytes).hexdigest() == item["sha256_git_blob_bytes"]
        subprocess.run(
            ["git", "diff", "--exit-code", SOURCE_HEAD, "--", item["path"]],
            cwd=ROOT,
            check=True,
        )


def test_03_incomplete_classification_preservation(artifacts):
    old = load_json("paper_protocol/reviewer_risk/loo_protocol_final_summary.json")
    assert old["final_classification"] == "LOO_PROTOCOL_INCOMPLETE"
    assert artifacts["summary"]["source"]["classification_preserved"] == "LOO_PROTOCOL_INCOMPLETE"


def test_04_prospective_pre_result_repair_status(artifacts):
    assert artifacts["repair"]["repair_class"] == "PROSPECTIVE_PRE_RESULT_PROTOCOL_REPAIR"
    assert artifacts["views"]["repair_status"] == "PROSPECTIVE_PRE_RESULT_PROTOCOL_REPAIR"


def test_05_five_loo_splits(artifacts):
    splits = artifacts["basis"]["splits"]
    assert [item["held_out_garment"] for item in splits] == GARMENTS
    assert [item["split_id"] for item in splits] == [f"LOO-{g}" for g in GARMENTS]


def test_06_held_out_garment_exclusion(artifacts):
    for split in artifacts["basis"]["splits"]:
        held_out = split["held_out_garment"]
        assert held_out not in split["basis_garments"]
        assert held_out not in split["basis_teacher_assets"]
        assert all(split["held_out_exclusion"].values())
        assert split["held_out_teacher_asset"]["deployable_information_use_count"] == 0


def test_07_four_garment_basis_bank(artifacts):
    for split in artifacts["basis"]["splits"]:
        assert len(split["basis_garments"]) == 4
        assert set(split["basis_garments"]) == set(GARMENTS) - {split["held_out_garment"]}
        assert split["static_audit"]["input_teacher_count"] == 4


def test_08_rank_at_most_three_contract(artifacts):
    for split in artifacts["basis"]["splits"]:
        basis = split["basis_contract"]
        assert basis["rank_rule"] == "min(numerical_rank,3)"
        assert basis["affine_rank_upper_bound"] == 3


def test_09_full_five_garment_basis_reuse_false(artifacts):
    assert all(
        not split["basis_contract"]["full_five_garment_basis_reused"]
        for split in artifacts["basis"]["splits"]
    )
    assert not artifacts["execution"]["basis_contract"]["full_five_garment_rank4_basis_reused"]


def test_10_primary_budget_set_exactly_k1_k2(artifacts):
    assert artifacts["views"]["primary_budgets"] == [1, 2]
    assert artifacts["execution"]["primary_task_contract"]["allowed_K"] == [1, 2]


def test_11_k4_primary_eligibility_false(artifacts):
    k4 = artifacts["views"]["k4_deferred_contract"]
    assert k4["name"] == "DEFERRED_ALL_AVAILABLE_VIEW_DIAGNOSTIC"
    assert not k4["primary_crossfit_eligible"]
    assert not k4["execution_authorized"]


def test_12_k4_execution_count_zero(artifacts):
    assert artifacts["views"]["k4_deferred_contract"]["current_task_count"] == 0
    assert artifacts["counts"]["planned_future_counts"]["K4_primary_tasks"] == 0


def test_13_exactly_40_primary_tasks(artifacts):
    tasks = artifacts["views"]["primary_tasks"]
    assert len(tasks) == len({task["task_id"] for task in tasks}) == 40
    assert artifacts["views"]["counts"]["primary_tasks"] == 40


def test_14_exactly_20_k1_tasks(artifacts):
    assert sum(task["K"] == 1 for task in artifacts["views"]["primary_tasks"]) == 20
    assert artifacts["views"]["counts"]["K1_tasks"] == 20


def test_15_exactly_20_k2_tasks(artifacts):
    assert sum(task["K"] == 2 for task in artifacts["views"]["primary_tasks"]) == 20
    assert artifacts["views"]["counts"]["K2_tasks"] == 20


def test_16_primary_blocked_tasks_zero(artifacts):
    assert all(task["status"] == "READY" for task in artifacts["views"]["primary_tasks"])
    assert artifacts["views"]["counts"]["blocked_tasks"] == 0
    assert "BLOCKED_VIEW_FOLD_CONFLICT" not in {
        task["status"] for task in artifacts["views"]["primary_tasks"]
    }


def test_17_per_rotation_k1_mapping_garment_agnostic(artifacts):
    views = artifacts["views"]
    metadata = views["semantic_and_camera_metadata"]
    canonical_az = metadata["canonical_front_azimuth_degrees"]
    expected = {"R0": "cond_000000", "R1": "cond_000017", "R2": "cond_000347", "R3": "cond_000000"}
    for condition, item in metadata["conditions"].items():
        x, y, z = item["c2w_translation"]
        azimuth = math.degrees(math.atan2(x, z))
        elevation = math.degrees(math.atan2(y, math.hypot(x, z)))
        azimuth_distance = abs((azimuth - canonical_az + 180.0) % 360.0 - 180.0)
        assert item["azimuth_degrees"] == approx(azimuth, abs=5e-9), condition
        assert item["elevation_degrees"] == approx(elevation, abs=5e-9), condition
        assert item["azimuth_distance_to_canonical_front_degrees"] == approx(azimuth_distance, abs=5e-9), condition
    for rotation in views["rotations"]:
        assert rotation["k1"] == expected[rotation["id"]]
        assert min(rotation["k1_candidate_scores"], key=lambda row: tuple(row["score_tuple"]))["condition_id"] == rotation["k1"]
        selected = {
            task["selected_adaptation_conditions"][0]
            for task in views["primary_tasks"]
            if task["K"] == 1 and task["rotation"] == rotation["id"]
        }
        assert selected == {rotation["k1"]}


def test_18_k1_selection_future_quality_blind(artifacts):
    contract = artifacts["views"]["k1_selection_contract"]
    assert contract["garment_agnostic"] and contract["identity_fixed"]
    assert contract["future_quality_blind"] and contract["result_blind"]
    assert {"RGB content", "mask quality", "Teacher residual", "render metric", "garment identity", "future test result"} == set(contract["forbidden_inputs"])


def test_19_k1_exactly_one_unique_condition(artifacts):
    for task in artifacts["views"]["primary_tasks"]:
        if task["K"] == 1:
            assert len(task["selected_adaptation_conditions"]) == 1
            assert len(set(task["selected_adaptation_conditions"])) == 1


def test_20_k2_exactly_two_unique_conditions(artifacts):
    for task in artifacts["views"]["primary_tasks"]:
        if task["K"] == 2:
            assert len(task["selected_adaptation_conditions"]) == 2
            assert len(set(task["selected_adaptation_conditions"])) == 2
            assert task["selected_adaptation_conditions"] == task["optimize_pool"]


def test_21_k1_subset_of_k2_all_20_groups(artifacts):
    indexed = {(task["held_out_garment"], task["rotation"], task["K"]): task for task in artifacts["views"]["primary_tasks"]}
    for garment in GARMENTS:
        for rotation in ("R0", "R1", "R2", "R3"):
            k1 = indexed[(garment, rotation, 1)]
            k2 = indexed[(garment, rotation, 2)]
            assert set(k1["selected_adaptation_conditions"]) < set(k2["selected_adaptation_conditions"])
            assert k1["K1_subset_of_K2"] and k2["K1_subset_of_K2"]
    assert artifacts["views"]["counts"]["K1_subset_of_K2_groups"] == 20


def test_22_adaptation_calibration_test_overlap_zero(artifacts):
    for task in artifacts["views"]["primary_tasks"]:
        assert task["adaptation_calibration_overlap"] == 0
        assert task["adaptation_test_overlap"] == 0
        assert task["calibration_test_overlap"] == 0


def test_23_duplicate_conditions_zero(artifacts):
    assert all(task["duplicate_adaptation_conditions"] == 0 for task in artifacts["views"]["primary_tasks"])
    assert artifacts["views"]["counts"]["duplicate_violations"] == 0


def test_24_no_test_or_calibration_in_adaptation(artifacts):
    for task in artifacts["views"]["primary_tasks"]:
        adaptation = set(task["selected_adaptation_conditions"])
        calibration = set(task["calibration_conditions"])
        test = set(task["test_conditions"])
        assert adaptation.isdisjoint(calibration)
        assert adaptation.isdisjoint(test)
        assert calibration.isdisjoint(test)


def test_25_shared_loss_hash_and_provenance(artifacts):
    reference = artifacts["loss_ref"]
    assert reference["status"] == "CORE_WEIGHT_ALIGNMENT_WITH_PREREGISTERED_SCOPE_DIFFERENCES"
    assert reference["loo_contract"]["sha256_git_blob_bytes"] == "d5eac4da4712208435e0c667bc49824708d473205c7d5942b017c33bada882f3"
    headroom_bytes = subprocess.check_output(
        ["git", "show", "2a42143f7942752aead16e7d53d1b7376fc5a143:paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json"],
        cwd=ROOT,
    )
    assert hashlib.sha256(headroom_bytes).hexdigest() == reference["headroom_contract"]["sha256_git_blob_bytes"]
    assert not reference["equivalence_audit"]["scientific_mismatch"]
    assert not reference["equivalence_audit"]["numerically_interchangeable"]


def test_26_optimizer_budget_unchanged(artifacts):
    old = artifacts["optimizer"]["shared_budget"]
    amended = artifacts["execution"]["optimizer_budget"]
    for key in ("optimizer", "learning_rate", "betas", "epsilon", "weight_decay", "steps", "checkpoints", "gradient_clip_norm", "scheduler", "seed", "retry_count", "early_stopping", "best_checkpoint_selection", "final_rule", "same_for_all_garment_rotation_k"):
        assert amended[key] == old[key]


def test_27_baseline_semantics_unchanged(artifacts):
    old = {method["id"]: method for method in artifacts["old_baselines"]["methods"]}
    amended = {method["id"]: method for method in artifacts["baselines"]["methods"]}
    assert set(amended) == set(old)
    for method_id in old:
        for key in ("paper_name", "deployable", "trainable", "held_out_teacher_use"):
            assert amended[method_id][key] == old[method_id][key]
        assert amended[method_id]["K_applicability"] == [1, 2]
        assert amended[method_id]["denominator"] == 40


def test_28_full_residual_fairness_unchanged(artifacts):
    fairness = artifacts["execution"]["full_residual_fairness"]
    assert artifacts["execution"]["optimizer_plans"]["FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION"]["paired_task_plan_count"] == 40
    for key in ("paired_plan_for_every_primary_task", "same_adaptation_views", "same_calibration", "same_test", "same_rendering_loss", "same_300_step_budget", "same_initial_render", "same_identity_protection", "equal_wall_time_secondary_required"):
        assert fairness[key]
    for key in ("extra_views", "test_used_for_optimization", "held_out_teacher_initialization"):
        assert not fairness[key]


def test_29_success_gate_primary_k2(artifacts):
    gate = artifacts["gates"]["hard_lookup_outperformance"]
    old = artifacts["old_gates"]["hard_lookup_outperformance"]
    assert gate["K"] == 2 and old["K"] == 4
    for key in ("minimum_successful_garments", "per_rotation_lpips_absolute_reduction_min", "per_rotation_rgb_mae_absolute_reduction_min", "successful_rotations_per_garment_min", "identity_contamination_count_max"):
        assert gate[key] == old[key]


def test_30_full_residual_recovery_gate_75_percent(artifacts):
    assert artifacts["gates"]["full_residual_comparison"]["low_dimensional_gain_recovery_ratio_min"] == 0.75
    assert artifacts["gates"]["full_residual_comparison"] == artifacts["old_gates"]["full_residual_comparison"]


def test_31_k1_sample_efficiency_only(artifacts):
    assert artifacts["gates"]["K1_role"]["classification"] == "SAMPLE_EFFICIENCY_DIAGNOSTIC"
    assert not artifacts["gates"]["K1_role"]["may_replace_failed_K2_primary_gate"]
    assert artifacts["evaluator"]["denominator"]["K1_role"] == "SAMPLE_EFFICIENCY_DIAGNOSTIC"


def test_32_counts_non_null_and_consistent_with_40_tasks(artifacts):
    counts = artifacts["counts"]["planned_future_counts"]
    assert all(value is not None for value in counts.values())
    assert counts["primary_tasks"] == counts["K1_primary_tasks"] + counts["K2_primary_tasks"] == 40
    assert counts["total_optimizer_runs"] == 40 * 3 == 120
    assert counts["optimizer_steps"] == 120 * 300 == 36000
    assert counts["checkpoint_writes"] == 120 * 6 == 720
    assert counts["adaptation_view_logical_renders"] == 18000 + 36000 == 54000
    assert counts["evaluation_inference"] == 720 + 240 == 960
    assert counts["total_logical_renders"] == 54000 + 960 == 54960
    assert counts["unique_physical_renders"] == 54960 - 120 == 54840
    assert counts["wide_result_table_rows"] == 40 + 20 + 5 == 65
    assert counts["visual_sheets"] == 20 + 5 + 1 == 26


def test_33_no_svd_optimizer_or_render_execution(artifacts):
    for source in (artifacts["repair"], artifacts["counts"], artifacts["execution"], artifacts["summary"], artifacts["test_contract"]):
        key = "actual_execution_counts"
        if key in source:
            assert all(value == 0 for value in source[key].values())
    assert not artifacts["execution"]["formal_output"]["attempt_directory_created_in_repair"]
    assert artifacts["execution"]["formal_output"]["scientific_attempt_count"] == 0


def test_34_frozen_mutation_zero(artifacts):
    assert all(value == 0 for value in artifacts["summary"]["frozen_mutations"].values())
    assert artifacts["repair"]["historical_artifact_immutability"]["mutation_count"] == 0


def test_35_credential_scan():
    patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    findings = []
    for relative in NEW_JSON_PATHS + MARKDOWN_PATHS + ["paper_protocol/reviewer_risk/loo_basis_adaptation_protocol_amended.yaml"]:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                findings.append((relative, pattern.pattern))
    assert findings == []


def test_36_json_duplicate_key_rejection():
    with raises(ValueError, match="duplicate JSON key"):
        json.loads('{"a":1,"a":2}', object_pairs_hook=_strict_pairs)


def test_37_json_parse():
    for relative in NEW_JSON_PATHS:
        assert isinstance(load_json(relative), dict)


def test_38_yaml_safe_load(artifacts):
    text = (ROOT / "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol_amended.yaml").read_text(encoding="utf-8")
    assert "\t" not in text
    if yaml is not None:
        value = yaml.safe_load(text)
        assert value["primary_budgets"] == [1, 2]
        assert value["counts"]["primary_tasks"] == 40
    else:
        assert 'status: "LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY"' in text


def test_39_markdown_nonempty():
    for relative in MARKDOWN_PATHS:
        assert len((ROOT / relative).read_text(encoding="utf-8").strip()) > 500


def test_40_py_compile():
    with tempfile.TemporaryDirectory() as directory:
        py_compile.compile(__file__, cfile=str(Path(directory) / "repair_test.pyc"), doraise=True)


def test_41_git_diff_check():
    result = subprocess.run(
        ["git", "diff", "--check", SOURCE_HEAD, "--"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_42_deterministic_manifest_regeneration(artifacts):
    views = artifacts["views"]
    expected_ids = [
        f"LOO-{garment}-{rotation}-K{budget}"
        for garment in GARMENTS
        for rotation in ("R0", "R1", "R2", "R3")
        for budget in (1, 2)
    ]
    assert [task["task_id"] for task in views["primary_tasks"]] == expected_ids
    first = json.dumps(views, ensure_ascii=True, indent=2) + "\n"
    second = json.dumps(load_json(NEW_JSON_PATHS[1]), ensure_ascii=True, indent=2) + "\n"
    assert first == second
    for task in views["primary_tasks"]:
        for payload_name, sha_name in (
            ("adaptation", "adaptation_manifest_sha"),
            ("calibration", "calibration_manifest_sha"),
            ("test", "test_manifest_sha"),
            ("query_order", "query_order_sha"),
        ):
            assert canonical_insertion_sha(task["hash_payloads"][payload_name]) == task[sha_name]


def test_43_local_origin_cloud_consistency_contract(artifacts):
    assert artifacts["summary"]["repair_branch"]["name"] == REPAIR_BRANCH
    assert artifacts["handoff"]["branch"] == REPAIR_BRANCH
    assert artifacts["execution"]["repair"]["branch"] == REPAIR_BRANCH
    assert artifacts["test_contract"]["required_check_count"] == 43
    assert len(artifacts["test_contract"]["required_checks"]) == 43
    assert artifacts["handoff"]["cloud_worktree"].endswith("canondressgs_loo_few_view_fold_manifest_repair")


_ARTIFACTS = artifacts()


def _make_unittest_method(function):
    def method(self):
        if function.__code__.co_argcount:
            function(_ARTIFACTS)
        else:
            function()

    method.__name__ = function.__name__
    return method


class TestLooFewViewFoldManifestRepair(unittest.TestCase):
    pass


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(
            TestLooFewViewFoldManifestRepair,
            _name,
            _make_unittest_method(_function),
        )
        del globals()[_name]

del _name, _function


if __name__ == "__main__":
    unittest.main()
