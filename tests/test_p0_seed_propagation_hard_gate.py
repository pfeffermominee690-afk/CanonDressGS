from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml

from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl


ROOT = Path(__file__).resolve().parents[1]
AUDIT_TOOL = ROOT / "tools/paper/run_p0_seed_propagation_preflight.py"
FORMAL_REGISTRY = ROOT / "paper_protocol/experiment_registry.yaml"
REVIEWER_REGISTRY = (
    ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
)
SEED_AUDIT = ROOT / "paper_protocol/reviewer_risk/seed_propagation_audit.json"
POLICY_AUDIT = ROOT / "paper_protocol/reviewer_risk/initialization_policy_aware_audit.json"


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ours_v2_at_seed(seed: int) -> MultiOutfitLinearCoefficientControl:
    torch.manual_seed(seed)
    return MultiOutfitLinearCoefficientControl(16, 4)


def test_seed_changes_trainable_initialization() -> None:
    """The historical result and policy-aware interpretation must coexist."""
    fingerprints = {_state_sha(_ours_v2_at_seed(seed)) for seed in (0, 1, 2)}
    assert len(fingerprints) == 1
    historical = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_AUDIT.read_text(encoding="utf-8"))
    assert historical["status"] == "SEED-PROPAGATION-FAIL"
    assert policy["ours_v2"]["cross_seed_identity_interpretation"] == "EXPECTED_DETERMINISTIC_IDENTITY"
    assert policy["ours_v2"]["cross_seed_state_unique_count"] == 1
    assert policy["b6"]["cross_seed_state_unique_count"] == 3
    assert policy["m3_m4"]["cross_seed_trunk_unique_count"] == 3
    source = AUDIT_TOOL.read_text(encoding="utf-8")
    assert 'len(initialization_shas) == len(SEEDS)' in source
    assert '"SEED-PROPAGATION-FAIL"' in source


def test_same_seed_reproduces_initialization() -> None:
    assert _state_sha(_ours_v2_at_seed(2)) == _state_sha(_ours_v2_at_seed(2))


def test_seed_audit_records_failure_before_p0_optimizer() -> None:
    audit = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "SEED-PROPAGATION-FAIL"
    assert audit["failed_families"] == ["ours_v2"]
    assert audit["families"]["ours_v2"]["initialization_unique_count"] == 1
    assert audit["families"]["b6_reference_classifier"]["initialization_unique_count"] == 3
    assert audit["families"]["complex_fusion_corrected"]["initialization_unique_count"] == 3
    assert audit["p0_trainable_adapter_optimizer_created_before_gate"] is False
    assert audit["failure_action"]["stop_all_p0_training"] is True
    assert audit["failure_action"]["next_task"] == (
        "REPAIR_PAPER_SEED_PROTOCOL_AND_RERUN_TRAINABLE_GROUPS"
    )


def test_seed_gate_runs_before_any_optimizer() -> None:
    source = AUDIT_TOOL.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert '"optimizer_created": False' in source
    assert '"p0_trainable_adapter_optimizer_created_before_gate": False' in source


def test_formal_registry_is_unchanged() -> None:
    canonical_blob = subprocess.check_output(
        ["git", "show", "HEAD:paper_protocol/experiment_registry.yaml"], cwd=ROOT
    )
    assert hashlib.sha256(canonical_blob).hexdigest() == (
        "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"
    )


def test_strict_view_remains_blocked() -> None:
    registry = yaml.safe_load(REVIEWER_REGISTRY.read_text(encoding="utf-8"))
    row = next(
        item for item in registry["experiments"]
        if item["experiment_id"] == "RR-STRICT-VIEW-ONE-FOLD-CANARY"
    )
    assert row["status"] == "BLOCKED_PENDING_AUTHORIZATION"


def test_registry_encodes_seed_gate_failure_without_fake_runs() -> None:
    registry = yaml.safe_load(REVIEWER_REGISTRY.read_text(encoding="utf-8"))
    rows = {item["experiment_id"]: item for item in registry["experiments"]}
    assert registry["p0_hard_gate"]["status"] == "SEED-PROPAGATION-FAIL"
    assert registry["p0_hard_gate"]["p0_optimizer_steps"] == 0
    assert all(rows[f"RR-OURS-V2-S{seed}"]["status"] == "FAILED" for seed in range(3))
    assert all(
        rows[f"RR-B6-REFERENCE-CLASSIFIER-S{seed}"]["status"] == "PREFLIGHT_PASS"
        for seed in range(3)
    )
    assert all(
        rows[f"RR-M3-COMPLEX-CORRECTED-S{seed}"]["status"] == "PREFLIGHT_PASS"
        for seed in range(3)
    )
    assert rows["RR-B7-F2-NEAREST-CENTROID-FIXED"]["status"] == "NOT_RUN"


def test_no_paper_final_is_created() -> None:
    registry = REVIEWER_REGISTRY.read_text(encoding="utf-8")
    assert "PAPER_FINAL" not in registry
