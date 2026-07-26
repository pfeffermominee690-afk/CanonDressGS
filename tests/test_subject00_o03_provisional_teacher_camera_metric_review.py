from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TASK_ID = "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-METRIC-REVIEW-001"
SOURCE_HEAD = "bc0798e17df6dd2c6eccefde527973aa15ca1cc8"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_COMPLETED_CAMERA_CONTRACT_"
    "CONTAMINATED_REQUIRES_7VIEW_RERUN"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN"


FILES = {
    "view_usage": RISK
    / "subject00_O03_provisional_teacher_view_usage_audit_20260727.json",
    "camera": RISK
    / "subject00_O03_provisional_teacher_camera_contract_audit_20260727.json",
    "metrics": RISK
    / "subject00_O03_provisional_teacher_per_view_metrics_20260727.json",
    "discrepancy": RISK
    / "subject00_O03_provisional_teacher_metric_discrepancy_audit_20260727.json",
    "seven": RISK
    / "subject00_O03_provisional_teacher_7view_diagnostic_summary_20260727.json",
    "review": RISK
    / "subject00_O03_provisional_teacher_human_review_manifest_20260727.json",
    "tests": RISK
    / "subject00_O03_provisional_teacher_camera_metric_review_tests_20260727.json",
    "summary": RISK
    / "subject00_O03_provisional_teacher_camera_metric_review_final_summary_20260727.json",
    "handoff": ROOT
    / "project_control_handoff"
    / "subject00_O03_provisional_teacher_camera_metric_review_handoff_20260727.json",
}


def load(name: str):
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def test_required_artifacts_exist_and_parse():
    for path in FILES.values():
        assert path.is_file(), path
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value.get("task_id", value.get("TASK_ID")) == TASK_ID
    assert (
        RISK
        / "SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_METRIC_REVIEW_REPORT_20260727.md"
    ).is_file()
    assert (
        ROOT
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_METRIC_REVIEW_REPORT_20260727.md"
    ).is_file()


def test_source_run_and_sampling_authenticity():
    usage = load("view_usage")
    summary = load("summary")
    assert summary["SOURCE_HEAD"] == SOURCE_HEAD
    assert summary["TRAINING_STEP_COUNT"] == 1200
    assert summary["CHECKPOINT_COUNT"] == 5
    assert summary["CHECKPOINT_PARSE_STATUS"].startswith("PASS_5_OF_5")
    assert usage["target_cell_count"] == 8
    assert usage["target_camera_ids"] == [17, 21, 14, 23, 11, 2, 9, 5]
    assert usage["sampling"]["sample_count_sum"] == 1200
    assert usage["sampling"]["view_denominator_values"] == [8]
    assert usage["sampling"]["view_sample_counts"] == {
        f"slot_{index:02d}": 150 for index in range(8)
    }
    assert usage["sampling"]["slot04_included_in_optimizer"] is True
    assert usage["sampling"]["slot04_included_in_loss_denominator"] is True
    assert usage["sampling"]["slot04_included_in_final_metrics"] is True


def test_exact_target_inventory_and_asset_bindings():
    usage = load("view_usage")
    expected_requests = [
        "subject00_O03_slot00_canary_attempt004_cand00",
        "subject00_O03_slot01_remaining_attempt005_cand00",
        "subject00_O03_slot02_cand00",
        "subject00_O03_slot03_cand01",
        "subject00_O03_slot04_canary_attempt004_cand00",
        "subject00_O03_slot05_canary_attempt004_cand00",
        "subject00_O03_slot06_remaining_attempt005_cand00",
        "subject00_O03_slot07_canary_attempt004_cand00",
    ]
    assert usage["target_request_ids"] == expected_requests
    for index, row in enumerate(usage["target_inventory"]):
        assert row["sampler_index"] == index
        assert row["actual_sampled_count"] == 150
        assert row["raw"]["status"] == "PASS"
        assert row["person_mask"]["status"] == "PASS"
        assert row["garment_mask"]["status"] == "PASS"
        for key in ("raw", "person_mask", "garment_mask"):
            assert len(row[key]["sha256"]) == 64


def test_slot04_camera_contract_and_latest_quarantine():
    camera = load("camera")["camera_contract"]
    slot04 = camera["slot04"]
    latest = camera["latest_camera_blocker_resolution"]
    assert slot04["transform_model"] == "SIMILARITY"
    assert slot04["training_target_K"] is None
    assert slot04["formal_preflight_registered_source_to_target_similarity"] is None
    assert slot04["formal_preflight_target_K"] is None
    assert (
        slot04["formal_preflight_binding_status"]
        == "BLOCKED_HUMAN_OVERRIDE_DOES_NOT_SELECT_UNIQUE_PHYSICAL_CAMERA"
    )
    assert slot04["conflicts_with_formal_preflight_blocker"] is True
    assert latest["head"] == "a434ae7a78fe898be2658180f20bbcd4391a64c0"
    assert latest["status"] == (
        "SEALED_RESOLUTION_WITH_QUARANTINE_SLOT04_REMAINS_"
        "UNRESOLVED_REVIEW_ONLY"
    )
    assert latest["slot04"]["training_eligible"] is False
    assert latest["slot04"]["evaluation_eligible"] is False
    assert latest["slot04"]["review_only"] is True
    assert latest["slot04"]["selected_model"] is None


def test_checkpoint_and_parameter_authenticity():
    summary = load("summary")
    assert summary["FINAL_CHECKPOINT_SHA256"] == (
        "41755ab9925b91b5ed98e86b62e07fdb36b5ab31421dcbcd6d72347302000c21"
    )
    assert summary["INITIALIZATION_SHA256"] == (
        "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
    )
    assert summary["TRAINABLE_PARAMETER_CHANGE_STATUS"] == (
        "PASS_ALL_5_TRAINABLE_TENSORS_CHANGED_FROM_STEP0_TO_STEP1200"
    )
    assert summary["FROZEN_PARAMETER_MUTATION_STATUS"].startswith("PASS_")


def test_per_view_metrics_and_three_aggregates():
    metrics = load("metrics")
    rows = metrics["per_view"]
    assert len(rows) == 8
    assert [row["slot"] for row in rows] == [f"slot_{i:02d}" for i in range(8)]
    assert [row["camera_id"] for row in rows] == [17, 21, 14, 23, 11, 2, 9, 5]
    required = {
        "full_image_lpips",
        "psnr",
        "ssim",
        "garment_region_lpips",
        "garment_region_psnr",
        "garment_region_ssim",
        "silhouette_iou",
        "boundary_f",
        "protected_region_lpips",
        "protected_region_rgb_mae",
        "alpha_foreground_error",
        "severe_artifact_flag",
    }
    assert all(required <= set(row) for row in rows)
    assert metrics["full_8view_aggregate"]["view_count"] == 8
    assert (
        metrics["camera_safe_7view_aggregate_excluding_slot04"]["view_count"] == 7
    )
    assert metrics["slot04_only"]["view_count"] == 1
    assert all(row["severe_artifact_flag"] is False for row in rows)


def test_lpips_exact_recompute_and_contract():
    discrepancy = load("discrepancy")
    assert discrepancy["full_image_lpips_original"] == 0.7719147130846977
    assert discrepancy["full_image_lpips_recomputed"] == 0.7719147130846977
    assert discrepancy["garment_region_lpips_recomputed"] == 0.020961973117664456
    assert discrepancy["protected_region_lpips_recomputed"] == 0.009790473792236298
    assert discrepancy["metric_contract_issue"] is False
    assert discrepancy["correction_overlay_required"] is False
    assert discrepancy["metric_implementation_status"].startswith("PASS_")
    assert all(
        status.startswith("PASS_")
        for status in discrepancy["per_view_pairing_status"].values()
    )
    assert discrepancy["background_share_of_full_absolute_error_mean"] > 0.99


def test_7view_is_diagnostic_not_decontaminated():
    seven = load("seven")
    assert seven["aggregate"]["view_count"] == 7
    assert seven["checkpoint_contaminated_by_excluded_view"] is True
    assert seven["clean_7view_rerun_required"] is True
    assert "DIAGNOSTIC" in seven["semantics"]
    assert seven["final_classification"] == FINAL_CLASSIFICATION
    assert seven["next_task"] == NEXT_TASK


def test_review_package_and_fixed_human_fields():
    review = load("review")
    summary = load("summary")
    assert review["classification"] == "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW"
    assert len(review["files"]) == 12
    assert all(review["required_content"].values())
    assert review["human_visual_decision"] is None
    assert review["scientific_pass"] is None
    assert review["paper_eligible"] is False
    assert review["run_root_only_image_policy"] is True
    assert review["git_image_count"] == 0
    assert summary["HUMAN_VISUAL_DECISION"] is None
    assert summary["SCIENTIFIC_PASS"] is None
    assert summary["PAPER_ELIGIBLE"] is False
    assert summary["PAPER_FINAL"] is False


def test_formal_base_mutations_classification_and_next_task():
    summary = load("summary")
    assert summary["FORMAL_BASE_STATUS"] == "USER_AUTHORIZED_PAUSED"
    assert summary["FORMAL_BASE_DURABLE_RESUME_STEP"] == 60747
    assert summary["FORMAL_BASE_RESUME_READY"] is True
    assert summary["FORMAL_BASE_RESUME_AUTHORIZED"] is False
    assert summary["OPTIMIZER_STEPS_BY_THIS_TASK"] == 0
    assert summary["DATA_MUTATIONS"] == 0
    assert summary["TARGET_MUTATIONS"] == 0
    assert summary["MASK_MUTATIONS"] == 0
    assert summary["CHECKPOINT_MUTATIONS"] == 0
    assert summary["PAPER_MODIFICATIONS"] == 0
    assert summary["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION
    assert summary["NEXT_TASK"] == NEXT_TASK


def test_audit_runner_has_no_optimizer_or_training_call_path():
    path = (
        ROOT
        / "tools"
        / "second_identity"
        / "audit_subject00_o03_provisional_teacher_camera_metrics.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert not any(node.func.attr == "backward" for node in calls)
    assert not any(node.func.attr == "step" for node in calls)
    assert "torch.optim" not in source
    assert "capacity_loss(" not in source


def test_runtime_structured_checks_and_no_task_images_in_git():
    tests = load("tests")
    assert tests["runtime_tests"]["result"] == "PASS"
    assert tests["runtime_tests"]["pass_count"] == 37
    assert tests["runtime_tests"]["test_count"] == 37
    assert tests["required_checks_passed"] == 37
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    ).splitlines()
    assert not any(
        path.lower().endswith(".png")
        and "camera_metric_review_20260727" in path
        for path in tracked
    )
