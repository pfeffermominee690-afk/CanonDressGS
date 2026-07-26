from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "research"
    / "subject00_o03_provisional_teacher_camerasafe7_v1.json"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "second_identity"
    / "run_subject00_o03_provisional_teacher_camerasafe7.py"
)
MATERIALIZER_PATH = (
    ROOT
    / "tools"
    / "second_identity"
    / "materialize_subject00_o03_camerasafe7_artifacts.py"
)
TASK_ID = (
    "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-SAFE-7VIEW-RERUN-001"
)
SOURCE_HEAD = "d541edce4b7b1da7ce6d055275993c1faf10e45a"
CAMERA_HEAD = "a434ae7a78fe898be2658180f20bbcd4391a64c0"
MASK_HEAD = "4ed89d9ac5076d13fef1c2cad8fcf28e3d236ac0"
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


def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_task_source_and_evidence_heads():
    value = config()
    assert value["task_id"] == TASK_ID
    assert value["source_git"]["head"] == SOURCE_HEAD
    assert value["evidence"]["camera_resolution_head"] == CAMERA_HEAD
    assert value["evidence"]["mask_acceptance_head"] == MASK_HEAD
    assert value["paper_eligible"] is False


def test_exact_safe7_target_contract():
    targets = config()["targets"]
    assert targets["schema"] == "canondressgs.full_dataset.v1"
    assert targets["data_class"] == (
        "RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT"
    )
    assert targets["count"] == targets["denominator"] == 7
    assert targets["slots"] == SAFE_SLOTS
    assert targets["camera_ids"] == SAFE_CAMERAS
    assert targets["request_ids"] == SAFE_REQUESTS


def test_slot04_is_only_the_explicit_excluded_view():
    targets = config()["targets"]
    assert targets["excluded"] == {
        "count": 1,
        "slot": "slot_04",
        "camera_id": 11,
        "direction": "right",
        "request_id": EXCLUDED_REQUEST,
        "reason": "REVIEW_ONLY_CAMERA_QUARANTINED",
    }
    active_text = json.dumps(
        {
            "slots": targets["slots"],
            "cameras": targets["camera_ids"],
            "requests": targets["request_ids"],
        }
    )
    assert "slot_04" not in active_text
    assert "cam11" not in active_text
    assert EXCLUDED_REQUEST not in active_text


def test_base_and_historical_checkpoint_roles_are_frozen():
    value = config()
    assert value["base"]["step"] == 60747
    assert value["base"]["checkpoint_sha256"] == (
        "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
    )
    historical = value["historical_contaminated_run"]
    assert historical["checkpoint_sha256"] == (
        "41755ab9925b91b5ed98e86b62e07fdb36b5ab31421dcbcd6d72347302000c21"
    )
    assert historical["initialization_allowed"] is False
    assert historical["comparison_only"] is True


def test_teacher_loss_optimizer_seed_and_step_contract():
    teacher = config()["teacher"]
    assert teacher["loss_name"] == "CAPACITY_ORACLE_LOSS_V1"
    assert teacher["loss_weights"] == {
        "garment_rgb": 1.0,
        "alpha_foreground": 0.5,
        "new_silhouette_alpha": 1.0,
        "boundary_rgb": 0.25,
        "protected_rgb": 10.0,
        "protected_alpha": 5.0,
        "stability": 0.0001,
    }
    assert teacher["optimizer"] == {
        "class": "Adam",
        "geometry_lr": 0.001,
        "appearance_lr": 0.002,
        "gradient_clip_norm": 1.0,
    }
    assert teacher["scheduler"] is None
    assert teacher["seed"] == 20260718
    assert teacher["steps"] == 1200


def test_balanced_schedule_and_checkpoint_contract():
    teacher = config()["teacher"]
    assert teacher["expected_view_sample_counts"] == {
        "slot_00": 172,
        "slot_01": 172,
        "slot_02": 172,
        "slot_03": 171,
        "slot_05": 171,
        "slot_06": 171,
        "slot_07": 171,
    }
    assert sum(teacher["expected_view_sample_counts"].values()) == 1200
    assert teacher["checkpoint_steps"] == [0, 300, 600, 900, 1200]


def test_runner_is_valid_python_and_has_one_fixed_training_loop():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ranges = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    ]
    assert any(
        len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 1201
        for node in ranges
    )
    assert "targets[(step - 1) % 7]" in source
    assert "optimizer.step()" in source
    assert "early_stop" not in source


def test_historical_checkpoint_load_is_after_all_optimizer_steps():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    training_loop = source.index("for step in range(1, 1201):")
    optimizer_step = source.index("optimizer.step()", training_loop)
    post_training_gate = source.index("dict(view_counts) != EXPECTED_COUNTS")
    historical_load = source.index(
        "historical_payload = torch.load(historical_checkpoint"
    )
    assert training_loop < optimizer_step < post_training_gate < historical_load
    assert (
        'historical_contaminated_checkpoint_used_for_initialization": False'
        in source
    )


def test_runner_constructs_only_safe7_camera_dataset():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    function_start = source.index("def build_camera_items_safe7(")
    function_end = source.index("\ndef target_hashes(", function_start)
    body = source[function_start:function_end]
    assert "cam_ids=list(SAFE_CAMERAS)" in body
    assert "camera == EXCLUDED_CAMERA" in body
    assert '"excluded_camera_loaded": False' in body
    assert "teacher.build_camera_items(" not in source


def test_runner_seals_loader_and_derived_target_evidence():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "derived_target_registry.json" in source
    for key in (
        "raw_tensor_sha256",
        "person_mask_tensor_sha256",
        "garment_mask_tensor_sha256",
        "protected_mask_tensor_sha256",
        "boundary_mask_tensor_sha256",
        "camera_record_sha256",
    ):
        assert key in source
    assert '"target_loader_parse": "PASS_7_OF_7"' in source


def test_runner_has_no_retry_attempt2_or_paper_mutation_path():
    value = config()
    execution = value["execution"]
    mutations = value["mutations"]
    assert execution == {
        "run_count": 1,
        "automatic_retry": False,
        "multi_seed": False,
        "hyperparameter_sweep": False,
        "early_stop": False,
    }
    assert mutations["formal_base_resume_allowed"] is False
    assert mutations["source_data_mutations_allowed"] is False
    assert mutations["historical_run_mutations_allowed"] is False
    assert mutations["paper_modifications_allowed"] is False
    assert mutations["attempt_002_allowed"] is False
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'attempt_002_created": False' in source
    assert 'mkdir("attempt_002")' not in source
    assert "AnonymousSubmission2027.tex" not in source


def test_materializer_declares_all_required_git_artifacts_and_runtime_checks():
    source = MATERIALIZER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    required = [
        "subject00_O03_camerasafe7_target_binding_registry_20260727.json",
        "subject00_O03_camerasafe7_view_sampling_registry_20260727.json",
        "subject00_O03_camerasafe7_training_registry_20260727.json",
        "subject00_O03_camerasafe7_checkpoint_registry_20260727.json",
        "subject00_O03_camerasafe7_per_view_metrics_20260727.json",
        "subject00_O03_camerasafe7_comparative_metrics_20260727.json",
        "subject00_O03_camerasafe7_animation_audit_20260727.json",
        "subject00_O03_camerasafe7_human_review_manifest_20260727.json",
        "subject00_O03_camerasafe7_execution_tests_20260727.json",
        "subject00_O03_camerasafe7_final_summary_20260727.json",
        "SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md",
        "subject00_O03_camerasafe7_provisional_teacher_handoff_20260727.json",
        "AAAI27_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md",
    ]
    assert all(name in source for name in required)
    runtime_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check"
    ]
    assert len(runtime_checks) >= 48
