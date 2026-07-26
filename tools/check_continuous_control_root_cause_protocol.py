from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper_protocol/reviewer_risk/continuous_control_root_cause_protocol.yaml"
RUNNER = ROOT / "tools/paper/run_continuous_control_artifact_root_cause.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("continuous_control_root_cause_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load root-cause runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert len(protocol["design"]["unordered_pair_order"]) == 10
    assert list(protocol["design"]["alpha"]) == [0.25, 0.50, 0.75]
    labels = protocol["design"]["stable_pair_labels"]
    assert sum(value == "stable" for value in labels.values()) == 3
    assert sum(value == "unstable" for value in labels.values()) == 7
    assert protocol["residual_contract"]["new_render_count"] == 720
    assert protocol["design"]["full_interpolation"]["required_render_count"] == 440
    assert protocol["no_training_gate"] == {
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_created": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "teacher_mutation": 0,
        "basis_mutation": 0,
        "formal_output_mutation": 0,
        "paper_final_count": 0,
    }
    runner = load_runner()
    left = runner.GaussianClothingResiduals.from_dict({
        name: torch.zeros((2, 1), dtype=torch.float64) for name in runner.CHANNELS
    })
    right = runner.GaussianClothingResiduals.from_dict({
        name: torch.full((2, 1), 2.0, dtype=torch.float64) for name in runner.CHANNELS
    })
    selected = runner.combine_residuals(left, right, ("delta_xyz",), 0.25)
    assert torch.equal(selected.delta_xyz, torch.full((2, 1), 0.5, dtype=torch.float64))
    for name in runner.CHANNELS:
        if name != "delta_xyz":
            assert torch.equal(getattr(selected, name), torch.ones((2, 1), dtype=torch.float64))
    midpoint = runner.combine_residuals(left, right, ("delta_xyz",), 0.50)
    assert all(torch.equal(getattr(midpoint, name), torch.ones((2, 1), dtype=torch.float64)) for name in runner.CHANNELS)
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert 0.0 <= runner.exact_permutation_p(values, {0, 1, 2}) <= 1.0
    print("CONTINUOUS_CONTROL_ROOT_CAUSE_PROTOCOL=PASS")


if __name__ == "__main__":
    main()
