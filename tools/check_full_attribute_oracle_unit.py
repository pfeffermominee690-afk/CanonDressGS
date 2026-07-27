from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.oracle_outfit_dataset import ResumableDeterministicSampler
from utils.full_training_checkpoint_utils import load_full_training_checkpoint, save_full_training_checkpoint


class Base:
    def __init__(self, count: int = 24) -> None:
        self._xyz = torch.zeros(count, 3)
        self._scaling = torch.zeros(count, 3)
        self._rotation = torch.zeros(count, 4)
        self._rotation[:, 0] = 1
        self._opacity = torch.zeros(count, 1)
        self._sh0 = torch.zeros(count, 1, 3)
        self._shN = torch.zeros(count, 3, 3)


def _exercise(oracle, base, expected_count: int, anchor: bool) -> None:
    assert oracle.oracle_type == "representation_capacity_upper_bound"
    initial = oracle(base)
    assert initial.gaussian_residuals.delta_xyz.shape == (expected_count, 3)
    assert initial.gaussian_residuals.delta_sh0.shape == (expected_count, 1, 3)
    assert initial.gaussian_residuals.delta_shN.shape == (expected_count, 3, 3)
    assert torch.allclose(initial.geometry_gate.mean(), torch.tensor(0.05), atol=1e-6)
    assert torch.count_nonzero(initial.gaussian_residuals.delta_xyz) == 0
    oracle.configure_stage(3)
    with torch.no_grad():
        oracle.raw_xyz.fill_(10)
        oracle.raw_log_scaling.fill_(10)
        oracle.raw_rotvec.fill_(10)
        oracle.raw_opacity.fill_(10)
        oracle.raw_sh0.fill_(10)
        oracle.geometry_gate_logits.fill_(10)
        oracle.appearance_gate_logits.fill_(10)
    output = oracle(base)
    assert output.gaussian_residuals.delta_xyz.abs().max() <= 0.05 + 1e-6
    assert output.gaussian_residuals.delta_log_scaling.abs().max() <= 0.35 + 1e-6
    assert torch.linalg.vector_norm(output.raw_residuals.delta_rotvec, dim=-1).max() <= 0.2617993878 + 1e-6
    assert output.gaussian_residuals.delta_opacity_logit.abs().max() <= 2 + 1e-6
    assert output.gaussian_residuals.delta_sh0.abs().max() <= 0.25 + 1e-6
    assert torch.count_nonzero(output.gaussian_residuals.delta_shN) == 0
    loss = output.canonical_overrides.xyz.square().mean() + output.regularization.geometry_gate_sparsity
    loss.backward()
    assert oracle.raw_xyz.grad is not None and torch.isfinite(oracle.raw_xyz.grad).all()
    assert oracle.geometry_gate_logits.grad is not None and torch.isfinite(oracle.geometry_gate_logits.grad).all()
    if anchor:
        assert output.geometry_gate.shape[0] != output.gaussian_geometry_gate.shape[0]


def _checkpoint_roundtrip(oracle, base) -> None:
    oracle.configure_stage(3)
    groups, names = oracle.parameter_groups({
        "geometry_residuals": 1e-3, "appearance_residuals": 2e-3,
        "geometry_gate": 1e-3, "appearance_gate": 1e-3,
    })
    optimizer = torch.optim.Adam(groups)
    optimizer.zero_grad()
    oracle(base).canonical_overrides.xyz.sum().backward()
    optimizer.step()
    method = {"oracle_type": oracle.oracle_type, "stage": 3}
    data = {"split_fingerprint": "fixture", "sampler_state": None}
    before = oracle(base).canonical_overrides.xyz.detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        save_full_training_checkpoint(
            path, model=oracle, optimizer=optimizer, optimizer_group_names=names,
            scheduler=None, scaler=None, training_state={"global_step": 1},
            data_state=data, method_state=method,
        )
        with torch.no_grad():
            oracle.raw_xyz.add_(1)
        load_full_training_checkpoint(
            path, model=oracle, optimizer=optimizer, optimizer_group_names=names,
            scheduler=None, scaler=None, expected_method_state=method, expected_data_state=data,
        )
        assert torch.equal(before, oracle(base).canonical_overrides.xyz)


def main() -> None:
    torch.manual_seed(7)
    base = Base()
    gaussian = GaussianResidualOracle(base)
    _exercise(gaussian, base, 24, False)
    indices = torch.arange(24).reshape(24, 1) % 6
    weights = torch.ones(24, 1)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    anchor = AnchorResidualOracle(base, 6, indices, weights, graph_edges=edges)
    _exercise(anchor, base, 24, True)
    _checkpoint_roundtrip(GaussianResidualOracle(base), base)
    first = ResumableDeterministicSampler(9, seed=3)
    prefix = []
    iterator = iter(first)
    for _ in range(4):
        prefix.append(next(iterator))
    state = first.state_dict()
    suffix = list(iterator)
    restored = ResumableDeterministicSampler(9, seed=3)
    restored.load_state_dict(state)
    assert suffix == list(restored)
    assert sorted(prefix + suffix) == list(range(9))
    print("full attribute oracle unit checks: PASS")


if __name__ == "__main__":
    main()
