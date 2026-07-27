from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/subject00_canondressgs_method_base60747_v1.json"
REGISTRY = (
    ROOT
    / "paper_protocol/reviewer_risk/subject00_base60747_three_garment_teacher_registry_20260727.json"
)
RUNNER = ROOT / "tools/second_identity/run_subject00_base60747_method.py"


def test_three_teacher_registry_is_frozen_technical_pass_only() -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert value["teacher_garment_count"] == 3
    assert value["base"]["step"] == 60747
    assert value["base"]["formal_base_status"] == "USER_AUTHORIZED_PAUSED"
    assert value["base"]["resume_authorized"] is False
    assert value["formal_targets"]["training_record_count"] == 22
    assert value["formal_targets"]["quarantine_count"] == 2
    assert set(value["garments"]) == {"O01", "O03", "O04"}
    for garment in value["garments"].values():
        assert garment["technical_status"] == "TECHNICAL_PASS"
        assert garment["human_status"] is None
        assert garment["scientific_pass"] is None
        assert garment["paper_eligible"] is False
        assert garment["steps"] == 1200
        assert garment["seed"] == 20260718
        assert garment["loss_contract"] == "CAPACITY_ORACLE_LOSS_V1"


def test_method_is_exact_pure_endpoint_contract() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    method = value["method_contract"]
    assert method["status"] == "UNIQUE_RECOVERED_FROM_PASSED_SUBJECT02_MAINLINE"
    assert method["precedent_head"] == "9ca0f44bdd5480a429e8cd5705df1341746d4381"
    assert method["canonical_implementation_lf_sha256"] == (
        "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6"
    )
    assert method["dual_support_policy"] == "DISABLED_EXCLUDED_FROM_PURE_ENDPOINT_PROTOCOL"
    assert value["controller"]["architecture"] == "LayerNorm(512) -> Linear(512,2); no activation"
    assert value["controller"]["parameter_count"] == 2050
    optimization = value["optimization"]
    assert optimization["optimizer"] == "torch.optim.Adam"
    assert optimization["learning_rate"] == 0.02
    assert optimization["steps_per_run"] == 300
    assert optimization["checkpoint_steps"] == [0, 20, 50, 100, 200, 300]
    assert optimization["initial_stability_step"] == 200
    assert optimization["loss"] == {
        "coefficient_smooth_l1_beta": 1.0,
        "coefficient_weight": 1.0,
        "classification_weight": 0.0,
        "pairwise_geometry_weight": 0.0,
        "render_weight": 0.0,
    }
    assert value["paper_eligible"] is False
    assert value["paper_final"] is False


def test_runner_enforces_bindings_and_no_render_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        "BASE_SHA",
        "MANIFEST_SHA",
        "F2_SHA",
        "F2_FINGERPRINT",
        "require_idle_gpu",
        "quarantine_count_in_manifest",
        "build_svd_basis",
        "FullDressableTrainingDataset",
        "FrozenF2ReferenceFeatureExtractor",
        "MultiOutfitLinearCoefficientControl",
        "smooth_l1_loss",
        "initial_stability_step200.json",
        "immutable_post_audit",
    ):
        assert required in source
    assert "render_direct(" not in source
    assert "DualSupport" not in source
    assert "ReferenceConditionedDualSupportController" not in source
