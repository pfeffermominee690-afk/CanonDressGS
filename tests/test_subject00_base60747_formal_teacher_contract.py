import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/subject00_base60747_three_garment_teachers_v1.json"
RUNNER = ROOT / "tools/second_identity/run_subject00_base60747_formal_teacher.py"


def test_three_garment_teacher_contract_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["task_id"] == "AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001"
    assert config["base"]["step"] == 60747
    assert config["base"]["checkpoint_sha256"] == "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
    assert config["base"]["formal_base_status"] == "USER_AUTHORIZED_PAUSED"
    assert config["base"]["resume_authorized"] is False
    assert config["formal_targets"]["training_record_count"] == 22
    assert config["formal_targets"]["quarantine_count"] == 2
    assert {name: row["target_count"] for name, row in config["garments"].items()} == {
        "O01": 7, "O03": 7, "O04": 8,
    }
    assert config["teacher"]["loss_name"] == "CAPACITY_ORACLE_LOSS_V1"
    assert config["teacher"]["steps"] == 1200
    assert config["teacher"]["seed"] == 20260718
    assert config["teacher"]["optimizer"] == {
        "class": "Adam",
        "geometry_lr": 0.001,
        "appearance_lr": 0.002,
        "gradient_clip_norm": 1.0,
    }
    assert config["teacher"]["scheduler"] is None
    assert config["teacher"]["checkpoint_steps"] == [0, 300, 600, 900, 1200]


def test_runner_has_integrity_and_quarantine_gates() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        "cpu_checkpoint_audit",
        "excluded_request_count_in_samples",
        "quarantine_sample_count",
        "base_fingerprint_unchanged",
        "formal_target_assets_unchanged",
        "different_camera_status",
        "different_pose_status",
        "garment_region_lpips",
        "silhouette_iou",
        "boundary_f",
        "protected_region_lpips",
        "protected_region_rgb_mae",
        "alpha_foreground_error",
        "scientific_pass",
        "paper_eligible",
    ):
        assert required in source
