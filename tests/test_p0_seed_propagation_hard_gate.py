from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl


ROOT = Path(__file__).resolve().parents[1]
AUDIT_TOOL = ROOT / "tools/paper/run_p0_seed_propagation_preflight.py"
FORMAL_REGISTRY = ROOT / "paper_protocol/experiment_registry.yaml"
REVIEWER_REGISTRY = (
    ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
)


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
    """The hard gate must expose, rather than hide, the current seed defect."""
    fingerprints = {_state_sha(_ours_v2_at_seed(seed)) for seed in (0, 1, 2)}
    assert len(fingerprints) == 1
    source = AUDIT_TOOL.read_text(encoding="utf-8")
    assert 'len(initialization_shas) == len(SEEDS)' in source
    assert '"SEED-PROPAGATION-FAIL"' in source


def test_same_seed_reproduces_initialization() -> None:
    assert _state_sha(_ours_v2_at_seed(2)) == _state_sha(_ours_v2_at_seed(2))


def test_seed_gate_runs_before_any_optimizer() -> None:
    source = AUDIT_TOOL.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert '"optimizer_created": False' in source
    assert '"optimizer_created_anywhere_in_audit": False' in source


def test_formal_registry_is_unchanged() -> None:
    assert hashlib.sha256(FORMAL_REGISTRY.read_bytes()).hexdigest() == (
        "dda4678657493d2a559dc3b6f3ca48371c24962ec426076e8343f98a5ad8c4c2"
    )


def test_strict_view_remains_blocked() -> None:
    registry = yaml.safe_load(REVIEWER_REGISTRY.read_text(encoding="utf-8"))
    row = next(
        item for item in registry["experiments"]
        if item["experiment_id"] == "RR-STRICT-VIEW-ONE-FOLD-CANARY"
    )
    assert row["status"] == "BLOCKED_PENDING_AUTHORIZATION"


def test_no_paper_final_is_created() -> None:
    registry = REVIEWER_REGISTRY.read_text(encoding="utf-8")
    assert "PAPER_FINAL" not in registry
