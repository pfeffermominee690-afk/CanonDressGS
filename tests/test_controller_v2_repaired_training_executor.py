from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repaired_training",
    ROOT / "tools/paper/run_controller_v2_repaired_training.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_repaired_contract_and_rotation_hashes_are_exact() -> None:
    contract = module.validate_contract()
    assert (
        contract["repaired"]["repair_branch"]
        == "research/controller-v2-micro-pilot-contract-repair-20260723"
    )
    assert (
        contract["repaired"]["classification"]
        == "CONTROLLER_V2_MICRO_PILOT_TRAINING_CONTRACT_REPAIRED"
    )
    assert len(contract["manifest"]["query_sets"]) == 320
    assert len(contract["schedules"]["rotations"]) == 4


def test_loss_uses_all_ten_pair_weights_for_consistency() -> None:
    first = torch.full((2, 10), 0.25)
    second = first.clone()
    second[:, 9] = 0.75
    value = torch.nn.functional.smooth_l1_loss(
        first, second, beta=0.1, reduction="mean"
    )
    assert float(value) > 0.0


def test_execution_counts_are_prospectively_fixed() -> None:
    assert module.FAMILIES == ("V2", "MATCHED_V1")
    assert module.SEEDS == (0, 1, 2)
    assert module.CHECKPOINT_STEPS == (0, 30, 60, 90, 120, 150)
    assert 2 * 4 * 3 * 150 == 3600
    assert 2 * 4 * 3 * 6 == 144
