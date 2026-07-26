from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TASK_ID = (
    "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-SAFE-7VIEW-RERUN-001"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_CAMERA_SAFE_7VIEW_"
    "TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW"
)
NEXT_TASK = "USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_RESULTS"
SAFE_SLOTS = [
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
]
SAFE_CAMERAS = [17, 21, 14, 23, 2, 9, 5]
SAFE_REQUESTS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
FILES = {
    "target": RISK
    / "subject00_O03_camerasafe7_target_binding_registry_20260727.json",
    "sampling": RISK
    / "subject00_O03_camerasafe7_view_sampling_registry_20260727.json",
    "training": RISK
    / "subject00_O03_camerasafe7_training_registry_20260727.json",
    "checkpoint": RISK
    / "subject00_O03_camerasafe7_checkpoint_registry_20260727.json",
    "metrics": RISK
    / "subject00_O03_camerasafe7_per_view_metrics_20260727.json",
    "comparative": RISK
    / "subject00_O03_camerasafe7_comparative_metrics_20260727.json",
    "animation": RISK
    / "subject00_O03_camerasafe7_animation_audit_20260727.json",
    "review": RISK
    / "subject00_O03_camerasafe7_human_review_manifest_20260727.json",
    "tests": RISK
    / "subject00_O03_camerasafe7_execution_tests_20260727.json",
    "summary": RISK
    / "subject00_O03_camerasafe7_final_summary_20260727.json",
    "handoff": ROOT
    / "project_control_handoff"
    / "subject00_O03_camerasafe7_provisional_teacher_handoff_20260727.json",
}


def load(name: str):
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def test_all_thirteen_required_git_artifacts_exist():
    for path in FILES.values():
        assert path.is_file(), path
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value.get("task_id", value.get("TASK_ID")) == TASK_ID
    assert (
        RISK / "SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md"
    ).is_file()
    assert (
        ROOT
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md"
    ).is_file()


def test_exact_safe7_target_and_asset_bindings():
    target = load("target")
    assert target["record_count"] == target["denominator"] == 7
    assert target["schema"] == "canondressgs.full_dataset.v1"
    assert target["target_data_class"] == (
        "RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT"
    )
    assert target["request_ids"] == SAFE_REQUESTS
    assert target["camera_ids"] == SAFE_CAMERAS
    assert target["slots"] == SAFE_SLOTS
    assert target["raw_binding_status"] == "PASS_7_OF_7_SHA256"
    assert target["person_mask_binding_status"] == "PASS_7_OF_7_SHA256"
    assert target["garment_mask_binding_status"] == "PASS_7_OF_7_SHA256"
    assert target["camera_binding_status"].startswith("PASS_7_OF_7")


def test_slot04_is_absent_from_every_active_target_path():
    target = load("target")
    active = {
        key: target[key]
        for key in ("request_ids", "camera_ids", "slots", "records")
    }
    text = json.dumps(active, sort_keys=True)
    assert "slot_04" not in text
    assert "cam11" not in text
    assert EXCLUDED_REQUEST not in text
    assert target["excluded_registry"]["records"][0]["request_id"] == EXCLUDED_REQUEST


def test_balanced_sampler_is_exact_and_slot04_zero():
    sampling = load("sampling")
    assert sampling["view_order"] == SAFE_SLOTS
    assert sampling["view_sample_counts"] == {
        "slot_00": 172,
        "slot_01": 172,
        "slot_02": 172,
        "slot_03": 171,
        "slot_05": 171,
        "slot_06": 171,
        "slot_07": 171,
    }
    assert sampling["sample_count_sum"] == 1200
    assert sampling["denominator_values"] == [7]
    assert sampling["slot04_sample_count"] == 0
    assert sampling["slot04_present_in_sampler"] is False


def test_training_contract_and_full_step_completion():
    training = load("training")
    assert training["loss_contract"] == "CAPACITY_ORACLE_LOSS_V1"
    assert training["optimizer"]["class"] == "Adam"
    assert training["scheduler"] is None
    assert training["seed"] == 20260718
    assert training["training_steps"] == 1200
    assert training["optimizer_steps_completed"] == 1200
    assert training["state_record_count"] == 1200
    assert training["checkpoint_events"] == [
        "step_000000.pth",
        "step_000300.pth",
        "step_000600.pth",
        "step_000900.pth",
        "step_001200.pth",
    ]
    assert training["automatic_retry"] is False
    assert training["attempt_002_created"] is False


def test_checkpoint_set_and_initialization_are_authentic():
    checkpoints = load("checkpoint")
    assert checkpoints["checkpoint_count"] == 5
    assert checkpoints["checkpoint_parse_status"] == "PASS_5_OF_5_CPU_TORCH_LOAD"
    assert [row["step"] for row in checkpoints["checkpoints"]] == [
        0,
        300,
        600,
        900,
        1200,
    ]
    assert checkpoints["initialization_sha256"] == (
        "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
    )
    assert checkpoints["exact_request_ids"] == SAFE_REQUESTS
    assert checkpoints["historical_checkpoint_used_for_initialization"] is False


def test_per_view_and_macro_metrics_use_denominator_seven():
    metrics = load("metrics")
    assert metrics["target_count"] == metrics["denominator"] == 7
    assert len(metrics["per_view"]) == 7
    assert metrics["macro"]["view_count"] == metrics["macro"]["denominator"] == 7
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
        "render_seconds",
    }
    assert all(required <= set(row) for row in metrics["per_view"])
    assert all(row["slot"] != "slot_04" for row in metrics["per_view"])


def test_three_way_comparison_is_safe7_and_historical_is_diagnostic_only():
    comparative = load("comparative")
    assert comparative["denominator"] == 7
    assert comparative["request_ids"] == SAFE_REQUESTS
    assert comparative["historical_run_role"] == (
        "HISTORICAL_DIAGNOSTIC_ONLY_NOT_INITIALIZATION"
    )
    assert comparative["base60747"]["macro"]["denominator"] == 7
    assert (
        comparative["historical_contaminated8_on_safe7"]["macro"]["denominator"]
        == 7
    )
    assert comparative["clean_camerasafe7"]["macro"]["denominator"] == 7
    assert len(comparative["clean_minus_contaminated_per_view"]) == 7


def test_animation_and_review_package_contracts():
    animation = load("animation")
    review = load("review")
    assert animation["deformation_status"] == "PASS_FINITE"
    assert animation["lbs_status"] == "PASS"
    assert animation["interpretation_boundary"].startswith("Finite-render")
    assert review["image_count"] == 10
    assert all(review["required_content"].values())
    assert review["run_root_only_image_policy"] is True
    assert review["git_image_count"] == 0
    assert review["human_visual_decision"] is None
    assert review["scientific_pass"] is None
    assert review["paper_eligible"] is False


def test_runtime_execution_checks_are_all_real_and_passing():
    tests = load("tests")
    assert tests["test_count"] >= 48
    assert tests["pass_count"] == tests["test_count"]
    assert tests["fail_count"] == 0
    assert tests["result"] == (
        f"PASS_RUNTIME_{tests['test_count']}_OF_{tests['test_count']}"
    )
    assert all(row["passed"] is True for row in tests["tests"])
    assert len({row["name"] for row in tests["tests"]}) == tests["test_count"]


def test_fixed_human_formal_base_mutation_and_decision_fields():
    summary = load("summary")
    assert summary["HUMAN_VISUAL_DECISION"] is None
    assert summary["SCIENTIFIC_PASS"] is None
    assert summary["PAPER_ELIGIBLE"] is False
    assert summary["FORMAL_BASE_STATUS"] == "USER_AUTHORIZED_PAUSED"
    assert summary["FORMAL_BASE_DURABLE_RESUME_STEP"] == 60747
    assert summary["FORMAL_BASE_RESUME_READY"] is True
    assert summary["FORMAL_BASE_RESUME_AUTHORIZED"] is False
    for key in (
        "OPTIMIZER_STEPS_BY_OTHER_TASKS",
        "DATA_MUTATIONS",
        "TARGET_MUTATIONS",
        "MASK_MUTATIONS",
        "BASE_CHECKPOINT_MUTATIONS",
        "CONTAMINATED_RUN_MUTATIONS",
        "PAPER_MODIFICATIONS",
    ):
        assert summary[key] == 0
    assert summary["PAPER_FINAL"] is False
    assert summary["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION
    assert summary["NEXT_TASK"] == NEXT_TASK


def test_no_review_images_are_committed_to_git():
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    ).splitlines()
    assert not any(
        path.lower().endswith((".png", ".jpg", ".jpeg"))
        and "camerasafe7" in path.lower()
        for path in tracked
    )
