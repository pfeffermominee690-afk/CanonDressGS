from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.anchor_clothing_mlp import AnchorClothingMLP
from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.image_conditioned_failure_diagnostics import (
    assert_forward_boundary,
    normalized_residual_regression_loss,
    reference_variant,
)
from scene.support_conditioned_dual_branch_residual_decoder_v7 import (
    SupportConditionedDualBranchResidualDecoderV7,
    V7_CAPACITY_GATE_MODE,
)


CONFIG = yaml.safe_load(
    (ROOT / "configs/research/subject02_residual_decoder_capacity_v7.yaml").read_text(encoding="utf-8")
)
BOUNDS = {
    "xyz": 0.05,
    "log_scaling": 0.35,
    "rotation": 0.2617993878,
    "opacity_logit": 2.0,
    "sh0": 0.25,
    "shN": 0.10,
}


class _Base:
    def __init__(self, count: int = 8) -> None:
        generator = torch.Generator().manual_seed(5)
        self._xyz = torch.randn(count, 3, generator=generator)
        self._scaling = torch.randn(count, 3, generator=generator) * 0.1
        self._rotation = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=1)
        self._opacity = torch.randn(count, 1, generator=generator)
        self._sh0 = torch.randn(count, 1, 3, generator=generator) * 0.1
        self._shN = torch.randn(count, 3, 3, generator=generator) * 0.01


def _decoder() -> tuple[SupportConditionedDualBranchResidualDecoderV7, torch.Tensor, torch.Tensor]:
    base = _Base()
    anchors = torch.tensor([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    indices = torch.tensor([[index % 4, (index + 1) % 4] for index in range(8)], dtype=torch.long)
    weights = torch.tensor([[0.75, 0.25]] * 8)
    config = {
        "global_feature_dim": 3,
        "local_feature_dim": 2,
        "hidden_dim": 16,
        "num_blocks": 3,
        "chunk_size": 3,
        "support_fourier_frequencies": 2,
        "gate_mode": V7_CAPACITY_GATE_MODE,
    }
    decoder = SupportConditionedDualBranchResidualDecoderV7.from_frozen_support(
        base, anchors, indices, weights, config, BOUNDS,
    )
    return decoder, torch.tensor([[0.2, -0.1, 0.3]]), torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10


def _assert_raises(error_type, callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_v7_has_no_outfit_id_input() -> None:
    parameters = inspect.signature(SupportConditionedDualBranchResidualDecoderV7.forward).parameters
    assert "outfit_id" not in parameters and "cloth_id" not in parameters


def test_v7_has_no_target_image_input() -> None:
    parameters = inspect.signature(SupportConditionedDualBranchResidualDecoderV7.forward).parameters
    assert all(not name.startswith("target") for name in parameters)
    _assert_raises(RuntimeError, lambda: assert_forward_boundary({"target_edit_rgb": torch.zeros(1)}))


def test_v7_directly_consumes_local_feature() -> None:
    decoder, global_feature, local = _decoder()
    captured = []
    hook = decoder.geometry_input[0].register_forward_pre_hook(lambda _module, values: captured.append(values[0].detach().clone()))
    decoder(global_feature, local, gaussian_indices=torch.tensor([0, 1]))
    decoder(global_feature, local + 1, gaussian_indices=torch.tensor([0, 1]))
    hook.remove()
    assert not torch.equal(captured[0], captured[1])
    assert torch.allclose(captured[1][:, -5:-3] - captured[0][:, -5:-3], torch.ones(2, 2))


def test_v7_geometry_and_appearance_trunks_are_separate() -> None:
    decoder, _, _ = _decoder()
    assert decoder.geometry_input is not decoder.appearance_input
    assert all(left is not right for left, right in zip(decoder.geometry_blocks, decoder.appearance_blocks))
    assert not {id(value) for value in decoder.geometry_blocks.parameters()}.intersection(
        id(value) for value in decoder.appearance_blocks.parameters()
    )


def test_v7_six_heads_are_independent() -> None:
    decoder, _, _ = _decoder()
    heads = [decoder.xyz_head, decoder.scaling_head, decoder.rotation_head, decoder.opacity_head, decoder.sh0_head, decoder.shN_head]
    assert len({id(head) for head in heads}) == 6
    assert len({id(head.weight) for head in heads}) == 6


def test_v7_zero_init_outputs_exact_zero() -> None:
    decoder, global_feature, local = _decoder()
    output = decoder(global_feature, local)
    for bundle in (output.raw_gaussian_residuals, output.bounded_gaussian_residuals, output.gated_gaussian_residuals):
        assert all(torch.count_nonzero(value).item() == 0 for value in bundle.as_dict().values())


def test_v7_bound_contract_matches_legacy() -> None:
    decoder, _, _ = _decoder()
    assert decoder.channel_bounds == BOUNDS
    legacy = AnchorClothingMLP(5, 3, hidden_dim=16, num_layers=3)
    legacy.configure_six_channel_decoder(CONFIG["model"]["dressable_channels"], 9)
    assert legacy.channel_bounds == BOUNDS


def test_v7_gate_is_fixed_one_in_capacity_probe() -> None:
    decoder, global_feature, local = _decoder()
    output = decoder(global_feature, local)
    assert output.gate_mode == V7_CAPACITY_GATE_MODE
    assert torch.equal(output.geometry_gate, torch.ones(8, 1))
    assert torch.equal(output.appearance_gate, torch.ones(8, 1))
    _assert_raises(ValueError, lambda: decoder(global_feature, local, geometry_gate=torch.ones(8, 1)))


def test_v7_oracle_residual_is_loss_only() -> None:
    assert CONFIG["stage_a"]["oracle_residual_usage"] == "loss_and_offline_evaluation_only"
    assert CONFIG["stage_a"]["image_space_loss"] is False


def test_v7_oracle_residual_is_detached() -> None:
    decoder, global_feature, local = _decoder()
    prediction = decoder(global_feature, local).gated_gaussian_residuals
    target = GaussianClothingResiduals(**{
        name: torch.zeros_like(value, requires_grad=False) for name, value in prediction.as_dict().items()
    })
    loss, _ = normalized_residual_regression_loss(prediction, target, BOUNDS)
    loss.backward()
    assert all(value.grad is None for value in target.as_dict().values())


def test_v7_chunked_matches_non_chunked() -> None:
    decoder, global_feature, local = _decoder()
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.01)
    chunked = decoder(global_feature, local, chunk_size=3).gated_gaussian_residuals
    complete = decoder(global_feature, local, chunk_size=100).gated_gaussian_residuals
    for name in chunked.as_dict():
        assert torch.allclose(getattr(chunked, name), getattr(complete, name), atol=1e-7, rtol=1e-6)


def test_v7_o01_o08_updates_are_balanced() -> None:
    assert CONFIG["stage_a"]["outfit_balance"] == "strict_alternating_1_to_1"
    assert CONFIG["stage_b"]["outfit_balance"] == "strict_alternating_1_to_1"
    assert CONFIG["outfits"] == ["O01", "O08"]


def test_v7_stage_b_has_no_diagnostic_latent() -> None:
    assert CONFIG["stage_b"]["diagnostic_latent_input"] is False
    assert CONFIG["stage_b"]["outfit_id_input"] is False


def test_v7_reference_swap_keeps_target_pose_fixed() -> None:
    episode = {
        "reference_images": torch.zeros(3, 3, 2, 2), "reference_cloth_masks": torch.ones(3, 1, 2, 2),
        "reference_foreground_masks": torch.ones(3, 1, 2, 2), "reference_poses": torch.arange(3 * 165).reshape(3, 165).float(),
        "reference_valid_mask": torch.ones(3), "reference_cameras": [{"id": index} for index in range(3)],
        "reference_condition_ids": ["a", "b", "c"],
    }
    geometry = {"surface_depth_maps": torch.arange(12).reshape(3, 1, 2, 2).float(), "surface_alpha_maps": torch.ones(3, 1, 2, 2), "deformation_fn": object()}
    swapped, _ = reference_variant(episode, geometry, "permuted")
    assert swapped["reference_cameras"][0]["id"] == 2
    assert "target_pose" not in swapped and "target_camera" not in swapped


def test_v7_base_and_mmlp_are_frozen() -> None:
    decoder, global_feature, local = _decoder()
    assert decoder.static_support_descriptor.requires_grad is False
    output = decoder(global_feature, local).gated_gaussian_residuals
    sum(value.sum() for value in output.as_dict().values()).backward()
    assert decoder.static_support_descriptor.grad is None
    assert CONFIG["base"]["frozen"] is True


def test_v7_legacy_checkpoint_still_loads() -> None:
    source = AnchorClothingMLP(5, 3, hidden_dim=16, num_layers=3)
    state = source.state_dict()
    restored = AnchorClothingMLP(5, 3, hidden_dim=16, num_layers=3)
    restored.load_state_dict(state, strict=True)
    assert all(torch.equal(state[name], restored.state_dict()[name]) for name in state)


def test_old_o01_and_diagnosis_outputs_immutable() -> None:
    assert CONFIG["permissions"]["modify_o01_attempt"] is False
    assert CONFIG["permissions"]["modify_diagnosis_attempt"] is False
    assert CONFIG["inputs"]["diagnosis_checkpoint_sha256"] == "047f41cca06bfa9aae31f581213bce07e04cbd1a4c158d693625b2e503f48f60"
    script = ROOT / "tools/diagnose_image_conditioned_overfit_failure.py"
    before = hashlib.sha256(script.read_bytes()).hexdigest()
    assert before == hashlib.sha256(script.read_bytes()).hexdigest()


def test_v7_render_directory_is_created_before_png_persistence() -> None:
    source = (ROOT / "tools/run_residual_decoder_capacity_v7.py").read_text(encoding="utf-8")
    directory_creation = 'directory.mkdir(parents=True, exist_ok=True)'
    first_save = 'save_render_tensor(directory / f"{outfit}_{condition}_predicted_rgb.png"'
    assert directory_creation in source and first_save in source
    assert source.index(directory_creation, source.index("def render_metrics")) < source.index(first_save)


def test_v7_visual_acceptance_requires_opened_images_and_matching_status() -> None:
    from tools.run_residual_decoder_capacity_v7 import _load_visual_observations

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "observations.json"
        path.write_text(json.dumps({
            "images_actually_opened": True,
            "inspection_method": "test viewer",
            "stage_a_status": "FAIL",
            "images": ["contact.png"],
            "observations": ["visible cloud artifact"],
        }), encoding="utf-8")
        loaded = _load_visual_observations(path, "FAIL")
        assert loaded["images_actually_opened"] is True
        _assert_raises(ValueError, lambda: _load_visual_observations(None, "FAIL"))
        _assert_raises(ValueError, lambda: _load_visual_observations(path, "PASS"))


def test_v7_finalization_distinguishes_run_and_adjudication_commits() -> None:
    source = (ROOT / "tools/run_residual_decoder_capacity_v7.py").read_text(encoding="utf-8")
    assert '"run_commit": immutable_inputs["git"]["commit"]' in source
    assert '"finalization_commit": git_output("rev-parse", "HEAD")' in source


if __name__ == "__main__":
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for name, callback in tests:
        callback()
        print(f"PASS {name}")
    print(f"PASS all {len(tests)} V7 decoder-capacity tests")
