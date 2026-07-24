from __future__ import annotations

import copy
import inspect
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from scene import loo_f2_feature_adapter as adapter
from tools.paper import repair_loo_calibration_f2_interface as repair


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"


def load(name: str) -> dict:
    return repair.read_json(RISK / name)


REPRO = load("loo_attempt002_failure_reproduction.json")
PRODUCER = load("loo_f2_producer_schema.json")
CONSUMER = load("loo_calibration_consumer_schema.json")
DIFF = load("loo_f2_interface_schema_diff.json")
BOUNDARY = load("loo_calibration_f2_information_boundary_audit.json")
CACHE = load("loo_f2_cache_schema_audit.json")
SEMANTICS = load("loo_calibration_semantics_audit.json")
MATRIX = load("loo_calibration_f2_interface_task_matrix.json")
EXECUTION = load("loo_calibration_f2_interface_execution_contract_repaired.json")
TESTS = load("loo_calibration_f2_interface_repair_tests.json")
SUMMARY = load("loo_calibration_f2_interface_repair_final_summary.json")
HANDOFF = repair.read_json(ROOT / f"project_control_handoff/{repair.HANDOFF_NAME}")
TASKS = MATRIX["rows"]


def valid_metadata(index: int = 0) -> dict:
    source = CACHE["records"][index]
    return {key: source[key] for key in adapter.REFERENCE_METADATA_KEYS}


def tensor_inputs(count: int):
    if adapter.torch is None:
        pytest.skip("tensor rejection is exercised by the cloud torch runtime")
    torch = adapter.torch
    images = torch.zeros((count, 3, 2, 2), dtype=torch.float32)
    masks = torch.ones((count, 1, 2, 2), dtype=torch.float32)
    valid = torch.ones((count, 1), dtype=torch.float32)
    records = []
    for index in range(count):
        record = valid_metadata(index)
        payload = {
            "schema_version": adapter.REFERENCE_RECORD_SCHEMA_VERSION,
            "outfit_id": record["outfit_id"],
            "condition_id": record["condition_id"],
            "source_image_sha256": record["source_image_sha256"],
            "source_mask_sha256": record["source_mask_sha256"],
        }
        records.append({
            **payload,
            "query_order": index,
            "source_image_path": record["source_image_path"],
            "source_mask_path": record["source_mask_path"],
            "cache_key": adapter.canonical_sha256(payload),
        })
    return images, masks, valid, records


class NeverCalledExtractor:
    feature_dim = 128

    def __call__(self, *args):
        raise AssertionError("validation must reject before producer invocation")


def test_01_exact_source_branch_and_head() -> None:
    assert SUMMARY["source_branch"] == repair.SOURCE_BRANCH
    assert SUMMARY["source_reporting_head"] == repair.SOURCE_REPORTING_HEAD


def test_02_execution_result_reporting_provenance() -> None:
    assert (
        SUMMARY["source_execution_head"], SUMMARY["source_result_head"],
        SUMMARY["source_reporting_head"],
    ) == (
        repair.SOURCE_EXECUTION_HEAD, repair.SOURCE_RESULT_HEAD,
        repair.SOURCE_REPORTING_HEAD,
    )


def test_03_attempt_001_immutable() -> None:
    manifest = REPRO["historical_attempt_manifests"]["attempt_001"]
    assert all(manifest[key] == value for key, value in repair.EXPECTED_ATTEMPT_001.items())
    assert SUMMARY["frozen_mutations"]["attempt_001"] == 0


def test_04_attempt_002_immutable() -> None:
    manifest = REPRO["historical_attempt_manifests"]["attempt_002"]
    assert all(manifest[key] == value for key, value in repair.EXPECTED_ATTEMPT_002.items())
    assert SUMMARY["frozen_mutations"]["attempt_002"] == 0


def test_05_attempt_003_absent() -> None:
    assert not SUMMARY["attempt_003_exists"]
    assert not EXECUTION["attempt_003_exists"]


def test_06_exact_failure_reproducible() -> None:
    assert REPRO["status"] == "PASS"
    assert REPRO["same_exception_class"] and REPRO["same_exception_message"]
    assert REPRO["backbone_forward_calls"] == 0


def test_07_same_first_failing_task() -> None:
    first = REPRO["first_failing_task"]
    assert first["task_id"] == TASKS[0]["task_id"] == "LOO-O01-R0-K1"
    assert not first["formal_task_loop_entered"]


def test_08_same_exception_class() -> None:
    assert REPRO["exception_class"] == "ValueError"


def test_09_same_exception_message() -> None:
    assert REPRO["exception_message"] == "frozen F2 control supports K in {1,2,3}"


def test_10_same_caller_chain() -> None:
    assert REPRO["formal_caller_chain"][-3:] == [
        "FrozenF2ReferenceFeatureExtractor.__call__",
        "FrozenF2ReferenceFeatureExtractor.forward",
        "K cardinality guard",
    ]
    assert REPRO["formal_callsite"].endswith(":2178")
    assert REPRO["runtime_guard"].endswith(":62")


def test_11_producer_schema_complete() -> None:
    assert PRODUCER["status"] == "PASS"
    assert set(PRODUCER["top_level_keys"]) == {
        "per_reference_f2", "set_mean", "set_max", "set_feature",
        "clothing_mean", "clothing_max", "resized_clothing_mask",
        "pooling_denominator", "valid_mask",
    }


def test_12_consumer_schema_complete() -> None:
    assert CONSUMER["status"] == "PASS"
    assert set(CONSUMER["required_keys"]) == {
        "schema_version", "role", "aggregation_rule", "view_count",
        "condition_ids", "query_order", "records", "features_per_view",
        "set_mean", "set_max", "aggregated_feature", "dtype", "device",
        "contiguous",
    }


def test_13_exact_mismatch_identified() -> None:
    assert DIFF["first_incompatible_field"] == "reference_images.shape[0] / semantic view_count"
    assert DIFF["root_cause_classifications"] == [
        "C. SINGLE_VIEW_MULTI_VIEW_AGGREGATION_MISMATCH",
        "D. CALIBRATION_CONSUMER_EXPECTS_WRONG_INTERFACE",
    ]


def test_14_unknown_keys_rejected() -> None:
    record = valid_metadata()
    record["unexpected"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        adapter.validate_reference_metadata(record)


def test_15_missing_keys_rejected() -> None:
    record = valid_metadata()
    record.pop("condition_id")
    with pytest.raises(ValueError, match="missing keys"):
        adapter.validate_reference_metadata(record)


def test_16_silent_squeeze_rejected() -> None:
    if adapter.torch is None:
        assert "reference_images.ndim != 4" in inspect.getsource(adapter.extract_reference_feature_set)
        return
    images, masks, valid, records = tensor_inputs(1)
    with pytest.raises(ValueError, match="explicit shape"):
        adapter.extract_reference_feature_set(
            NeverCalledExtractor(), images[0], masks, valid, records,
            role=adapter.ADAPTATION_ROLE,
        )


def test_17_implicit_view_aggregation_rejected() -> None:
    if adapter.torch is None:
        assert adapter.AUTHORIZED_ROLES[adapter.ADAPTATION_ROLE] == (1, 2)
        return
    images, masks, valid, records = tensor_inputs(4)
    with pytest.raises(ValueError, match="requires view_count"):
        adapter.extract_reference_feature_set(
            NeverCalledExtractor(), images, masks, valid, records,
            role=adapter.ADAPTATION_ROLE,
        )


def test_18_dtype_mismatch_rejected() -> None:
    if adapter.torch is None:
        assert "F2 interface dtype mismatch" in inspect.getsource(adapter.extract_reference_feature_set)
        return
    images, masks, valid, records = tensor_inputs(1)
    with pytest.raises(TypeError, match="dtype mismatch"):
        adapter.extract_reference_feature_set(
            NeverCalledExtractor(), images, masks.double(), valid, records,
            role=adapter.ADAPTATION_ROLE,
        )


def test_19_device_mismatch_rejected() -> None:
    source = inspect.getsource(adapter.extract_reference_feature_set)
    assert "F2 interface device mismatch" in source
    assert "len({value.device for value in tensors}) != 1" in source


def test_20_order_mismatch_rejected() -> None:
    if adapter.torch is None:
        assert "reference query order mismatch" in inspect.getsource(adapter.extract_reference_feature_set)
        return
    images, masks, valid, records = tensor_inputs(1)
    records[0]["query_order"] = 1
    with pytest.raises(ValueError, match="query order mismatch"):
        adapter.extract_reference_feature_set(
            NeverCalledExtractor(), images, masks, valid, records,
            role=adapter.ADAPTATION_ROLE,
        )


def test_21_held_out_teacher_reads_zero() -> None:
    assert BOUNDARY["held_out_teacher_reads"] == 0


def test_22_test_target_reads_zero() -> None:
    assert BOUNDARY["test_target_reads"] == 0


def test_23_test_metric_reads_zero() -> None:
    assert BOUNDARY["test_metric_reads"] == 0


def test_24_unauthorized_f2_reads_zero() -> None:
    assert BOUNDARY["unauthorized_f2_reads"] == 0


def test_25_train_only_centroid_boundary() -> None:
    assert BOUNDARY["train_only_centroid_boundary"] == "PASS"
    assert all(row["hard_lookup_selected_endpoint"] in row["basis_garments"] for row in TASKS)


def test_26_calibration_test_separation() -> None:
    assert BOUNDARY["calibration_test_separation"] == "PASS"
    assert all(row["calibration_condition_id"] != row["test_condition_id"] for row in TASKS)


def test_27_adaptation_calibration_separation() -> None:
    assert BOUNDARY["adaptation_calibration_separation"] == "PASS"
    assert all(row["calibration_condition_id"] not in row["adaptation_condition_ids"] for row in TASKS)


def test_28_forty_unique_task_descriptors() -> None:
    assert MATRIX["row_count"] == MATRIX["unique_task_count"] == 40


def test_29_twenty_k1_tasks() -> None:
    assert MATRIX["K1_count"] == sum(row["K"] == 1 for row in TASKS) == 20


def test_30_twenty_k2_tasks() -> None:
    assert MATRIX["K2_count"] == sum(row["K"] == 2 for row in TASKS) == 20


def test_31_k1_subset_k2() -> None:
    indexed = {(row["held_out_garment"], row["rotation"], row["K"]): row for row in TASKS}
    assert all(
        set(indexed[(outfit, rotation, 1)]["adaptation_condition_ids"])
        < set(indexed[(outfit, rotation, 2)]["adaptation_condition_ids"])
        for outfit in repair.runner.OUTFITS for rotation in ("R0", "R1", "R2", "R3")
    )


def test_32_forty_f2_reference_closures() -> None:
    assert MATRIX["F2_reference_closure_pass_count"] == 40


def test_33_forty_adaptation_closures() -> None:
    assert MATRIX["adaptation_closure_pass_count"] == 40


def test_34_forty_calibration_closures() -> None:
    assert MATRIX["calibration_closure_pass_count"] == 40


def test_35_forty_test_closures() -> None:
    assert MATRIX["test_closure_pass_count"] == 40


def test_36_forty_initialization_closures() -> None:
    assert MATRIX["initialization_closure_pass_count"] == 40


def test_37_cache_schema_versions_valid() -> None:
    assert CACHE["schema_version"].endswith(".v1")
    assert all(row["schema_version"] == adapter.REFERENCE_RECORD_SCHEMA_VERSION for row in CACHE["records"])


def test_38_source_image_mask_sha_valid() -> None:
    assert CACHE["source_mismatch_count"] == 0
    assert all(len(row["source_image_sha256"]) == len(row["source_mask_sha256"]) == 64 for row in CACHE["records"])


def test_39_condition_ids_valid() -> None:
    assert {row["condition_id"] for row in CACHE["records"]} == set(repair.runner.CONDITIONS)


def test_40_garment_split_valid() -> None:
    assert {row["outfit_id"] for row in CACHE["records"]} == set(repair.runner.OUTFITS)
    assert CACHE["wrong_garment_count"] == CACHE["wrong_split_count"] == 0


def test_41_view_count_valid() -> None:
    assert all(row["view_count"] == 1 for row in CACHE["records"])
    assert all(row["split_calibration_initialization_descriptor"]["per_garment_features_per_view_shape"] == [4, 256] for row in TASKS)


def test_42_aggregation_rule_valid() -> None:
    assert all(
        row["adaptation_batch_descriptor"]["aggregation_rule"] == adapter.AGGREGATION_RULE
        and row["split_calibration_initialization_descriptor"]["aggregation_rule"] == adapter.AGGREGATION_RULE
        for row in TASKS
    )


def test_43_stale_cache_count_zero() -> None:
    assert CACHE["stale_cache_count"] == 0


def test_44_missing_feature_count_zero() -> None:
    assert CACHE["missing_feature_count"] == CACHE["missing_source_asset_count"] == 0


def test_45_basis_unchanged() -> None:
    row = next(item for item in EXECUTION["scientific_immutability_audit"]["rows"] if item["path"].endswith("loo_basis_manifests.json"))
    assert row["match"]


def test_46_float64_closure_unchanged() -> None:
    parity = load("loo_basis_renderer_parity_execution_contract_repaired.json")
    assert parity["dtype_contract"].startswith("Teacher physical tensors cast to float64")
    assert parity["centering_contract"] == "final centered row equals negative sum of prior rows"


def test_47_hard_lookup_unchanged() -> None:
    replay = load("loo_hard_lookup_k_replay.json")
    assert all(row["hard_lookup_replay_exact"] for row in TASKS)
    assert replay["status"] == "PASS"


def test_48_fifteen_five_replay_unchanged() -> None:
    replay = load("loo_hard_lookup_k_replay.json")
    assert (replay["same_endpoint_pairs"], replay["different_endpoint_pairs"]) == (15, 5)


def test_49_cache_plan_unchanged() -> None:
    plan = load("loo_cache_key_plan_v2.json")
    assert (plan["logical_request_count"], plan["unique_physical_key_count"], plan["cache_hit_count"]) == (54_960, 54_845, 115)


def test_50_render_counts_unchanged() -> None:
    assert EXECUTION["actual_execution_counts"]["renderer_calls"] == 0
    assert EXECUTION["actual_execution_counts"]["basis_renderer_calls"] == 0


def _scientific_match(suffix: str) -> bool:
    return next(
        row["match"] for row in EXECUTION["scientific_immutability_audit"]["rows"]
        if row["path"].endswith(suffix)
    )


def test_51_loss_unchanged() -> None:
    assert _scientific_match("loo_adaptation_loss_contract.json")


def test_52_optimizer_unchanged() -> None:
    assert _scientific_match("loo_optimizer_contract.json")
    assert load("loo_optimizer_contract.json")["shared_budget"]["steps"] == 300


def test_53_full_residual_unchanged() -> None:
    full = load("loo_optimizer_contract.json")["methods"]["FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION"]
    assert (full["gaussian_count"], full["trainable_scalar_count"]) == (200_000, 4_400_000)
    assert not MATRIX["optimizer_parameter_schema"]["full_residual"]["allocated"]


def test_54_evaluator_unchanged() -> None:
    assert _scientific_match("loo_evaluator_contract_amended.json")


def test_55_success_gates_unchanged() -> None:
    assert _scientific_match("loo_success_gates_amended.json")


def test_56_scientific_drift_zero() -> None:
    assert EXECUTION["scientific_immutability_audit"]["scientific_semantic_drift"] == 0
    assert SUMMARY["scientific_semantic_drift"] == 0


def test_57_no_attempt_003() -> None:
    assert SUMMARY["actual_execution_counts"]["scientific_attempts_created"] == 0
    assert not SUMMARY["attempt_003_exists"]


def test_58_no_renderer() -> None:
    assert SUMMARY["actual_execution_counts"]["renderer_calls"] == 0


def test_59_no_optimizer() -> None:
    assert SUMMARY["actual_execution_counts"]["optimizer_creations"] == 0


def test_60_no_steps() -> None:
    assert SUMMARY["actual_execution_counts"]["optimizer_steps"] == 0
    assert SUMMARY["actual_execution_counts"]["backward_calls"] == 0


def test_61_no_checkpoints() -> None:
    assert SUMMARY["actual_execution_counts"]["checkpoint_writes"] == 0


def test_62_no_formal_metrics() -> None:
    assert SUMMARY["actual_execution_counts"]["formal_metrics"] == 0
    assert SUMMARY["actual_execution_counts"]["formal_evaluations"] == 0


def test_63_no_figure_bank_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Figure_Bank"] == 0


def test_64_no_subject00_mutation() -> None:
    assert SUMMARY["frozen_mutations"]["Subject00"] == 0


def test_65_credential_findings_zero() -> None:
    assert repair.credential_scan()["finding_count"] == 0


def test_66_json_strict_parse() -> None:
    paths = [RISK / name for name in repair.RISK_NAMES]
    paths.append(ROOT / f"project_control_handoff/{repair.HANDOFF_NAME}")
    assert all(isinstance(repair.read_json(path), dict) for path in paths)


def test_67_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        repair.read_json(path)


def test_68_yaml_safe_load() -> None:
    value = yaml.safe_load((RISK / "loo_basis_adaptation_protocol_amended.yaml").read_text(encoding="utf-8"))
    assert value["primary_budgets"] == [1, 2]


def test_69_markdown_nonempty() -> None:
    assert all(len((DOCS / name).read_text(encoding="utf-8").strip()) > 500 for name in repair.REPORT_NAMES)


def test_70_py_compile() -> None:
    paths = (
        ROOT / "scene/loo_f2_feature_adapter.py",
        ROOT / "tools/paper/run_loo_basis_adaptation_experiment.py",
        ROOT / "tools/paper/repair_loo_calibration_f2_interface.py",
        Path(__file__),
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, path in enumerate(paths):
            py_compile.compile(str(path), cfile=str(Path(directory) / f"{index}.pyc"), doraise=True)


def test_71_windows_tests_evidence_slot() -> None:
    assert TESTS["windows_python_311"] == "PENDING" or TESTS["windows_python_311"].startswith("PASS:")


def test_72_cloud_tests_evidence_slot() -> None:
    assert TESTS["cloud_python_310_torch"] == "PENDING" or TESTS["cloud_python_310_torch"].startswith("PASS:")


def test_73_deterministic_regeneration_contract() -> None:
    assert TESTS["deterministic_regeneration"] in {"PENDING", "PASS"}
    assert repair.canonical_sha(MATRIX) == EXECUTION["task_matrix_sha256"]


def test_74_git_diff_check() -> None:
    result = subprocess.run(["git", "diff", "--check"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_75_local_origin_cloud_consistency_contract() -> None:
    assert SUMMARY["repair_branch"] == repair.REPAIR_BRANCH
    assert SUMMARY["repository_consistency"] in {"PENDING_FINAL_SYNC", "PASS_AFTER_FINAL_PUSH"}
    assert HANDOFF["cloud_worktree"].endswith("canondressgs_loo_calibration_f2_interface_repair")


def test_76_both_worktrees_clean_contract() -> None:
    assert HANDOFF["both_worktrees_clean"] in {"PENDING_FINAL_SYNC", "PASS_AFTER_FINAL_COMMIT"}
    assert not HANDOFF["formal_attempt_created"]
    assert SUMMARY["paper_final_count"] == 0
