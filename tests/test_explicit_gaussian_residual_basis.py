from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from scene.explicit_gaussian_residual_basis import (
    CHANNEL_TO_BOUND,
    DiagnosticCoefficientPredictor,
    ExplicitGaussianResidualBasis,
    ReferenceBasisCoefficientPredictor,
    build_centered_difference_basis,
    build_svd_basis,
    normalized_residual_dict,
    project_residual_onto_basis,
)
from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from tools import run_explicit_gaussian_residual_basis as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/research/subject02_explicit_gaussian_residual_basis_v1.yaml").read_text(encoding="utf-8"))
BOUNDS = {"xyz": 0.05, "log_scaling": 0.35, "rotation": 0.25, "opacity_logit": 2.0, "sh0": 0.25, "shN": 0.1}


def _teacher(sign: float, count: int = 12) -> GaussianClothingResiduals:
    base = torch.linspace(-0.5, 0.5, count).reshape(count, 1) * sign
    return GaussianClothingResiduals(
        delta_xyz=base.repeat(1, 3) * BOUNDS["xyz"],
        delta_log_scaling=base.repeat(1, 3) * BOUNDS["log_scaling"],
        delta_rotvec=base.repeat(1, 3) * BOUNDS["rotation"],
        delta_opacity_logit=base * BOUNDS["opacity_logit"],
        delta_sh0=base.reshape(count, 1, 1).repeat(1, 1, 3) * BOUNDS["sh0"],
        delta_shN=base.reshape(count, 1, 1).repeat(1, 3, 3) * BOUNDS["shN"],
    )


def _decomposition():
    return build_centered_difference_basis({"O01": _teacher(-1), "O08": _teacher(1)}, BOUNDS, ("O01", "O08"))


def test_basis_is_explicit_per_gaussian_field() -> None:
    basis = _decomposition().basis
    assert isinstance(basis, ExplicitGaussianResidualBasis)
    assert basis.gaussian_count == 12 and basis.rank == 1
    assert basis.explicit_scalar_count == basis.mean_scalar_count + basis.basis_scalar_count
    assert not any(isinstance(module, torch.nn.Linear) for module in basis.modules())


def test_no_per_gaussian_decoder_in_basis_mode() -> None:
    assert CONFIG["model"]["decoder"]["type"] == "legacy"
    assert CONFIG["permissions"]["use_per_gaussian_residual_decoder"] is False
    source = inspect.getsource(runner.stage_c_prediction)
    assert "support_conditioned_residual_decoder_v7" not in source


def test_basis_is_bound_normalized() -> None:
    decomposition = _decomposition(); mean, components = decomposition.basis.normalized_fields()
    teachers = {"O01": _teacher(-1), "O08": _teacher(1)}
    expected = {name: normalized_residual_dict(value, BOUNDS) for name, value in teachers.items()}
    for name in CHANNELS:
        assert torch.equal(mean[name] - components[name][0], expected["O01"][name])
        assert torch.equal(mean[name] + components[name][0], expected["O08"][name])
        assert CHANNEL_TO_BOUND[name] in BOUNDS


def test_basis_composition_matches_teacher() -> None:
    decomposition = _decomposition()
    for outfit, target in (("O01", _teacher(-1)), ("O08", _teacher(1))):
        prediction = decomposition.basis(decomposition.teacher_coefficients[outfit])
        assert all(torch.equal(getattr(prediction, name), getattr(target, name)) for name in CHANNELS)


def test_basis_chunked_matches_non_chunked() -> None:
    decomposition = _decomposition(); coefficients = decomposition.teacher_coefficients["O08"]
    first, second = decomposition.basis(coefficients), decomposition.basis(coefficients, chunk_size=3)
    assert all(torch.equal(getattr(first, name), getattr(second, name)) for name in CHANNELS)


def test_teacher_not_loaded_during_prediction_forward() -> None:
    source = inspect.getsource(runner.stage_c_prediction) + inspect.getsource(runner.stage_c_coefficients)
    assert "_load_stage_a_teachers" not in source and "_load_oracle" not in source
    assert "teacher_coefficients" not in inspect.signature(runner.stage_c_prediction).parameters


def test_coefficient_predictor_outputs_only_coefficients() -> None:
    predictor = DiagnosticCoefficientPredictor(8, 2, 16)
    output = predictor(torch.randn(8))
    assert isinstance(output, torch.Tensor) and output.shape == (2,)


def test_stage_c_has_no_outfit_id() -> None:
    signature = inspect.signature(runner.stage_c_prediction)
    assert "outfit_id" not in signature.parameters and "cloth_id" not in signature.parameters
    assert CONFIG["permissions"]["outfit_id_in_model"] is False


def test_stage_c_has_no_diagnostic_latent() -> None:
    signature = inspect.signature(runner.stage_c_prediction)
    assert "diagnostic_latent" not in signature.parameters
    assert CONFIG["permissions"]["diagnostic_latent_in_stage_c"] is False


def test_stage_c_has_no_target_image_input() -> None:
    parameters = set(inspect.signature(runner.stage_c_prediction).parameters)
    assert not {"target_rgb", "target_mask", "target_image"}.intersection(parameters)
    assert CONFIG["permissions"]["target_image_in_prediction_forward"] is False


def test_reference_swap_keeps_target_pose_fixed() -> None:
    source = inspect.getsource(runner.evaluate_stage_c)
    assert 'context["geometries"][condition]' in source
    assert '"target_pose_camera_fixed_for_swap"] = True' in source


def test_o01_o08_updates_are_balanced() -> None:
    schedule = runner.balanced_episode_schedule(); counts = {item: 0 for item in schedule}
    for step in range(CONFIG["stage_c"]["max_steps"]): counts[schedule[step % len(schedule)]] += 1
    assert len(schedule) == 8 and set(counts.values()) == {62}
    assert sum(value for (outfit, _), value in counts.items() if outfit == "O01") == 248
    assert sum(value for (outfit, _), value in counts.items() if outfit == "O08") == 248


def test_basis_is_frozen_in_stage_c() -> None:
    basis = _decomposition().basis
    assert list(basis.parameters()) == []
    assert all(not value.requires_grad for value in basis.buffers())


def test_base_mmlp_renderer_are_frozen() -> None:
    for name in ("modify_base", "modify_mmlp", "modify_renderer"):
        assert CONFIG["permissions"][name] is False


def test_historical_outputs_are_immutable() -> None:
    assert CONFIG["permissions"]["modify_parameterization_attempt"] is False
    assert CONFIG["permissions"]["modify_oracle_outputs"] is False
    source = inspect.getsource(runner.finalize)
    assert "parameterization_outputs_immutable" in source


def test_reference_coefficient_predictor_uses_global_and_pooled_local() -> None:
    predictor = ReferenceBasisCoefficientPredictor(6, 4, 1, 12)
    output = predictor(torch.randn(1, 6), torch.randn(10, 4), torch.rand(10, 1))
    assert output.coefficients.shape == (1,)
    assert output.pooled_local_feature.shape == (1, 4)
    assert output.fused_feature.shape == (1, 10)


def test_coefficient_predictor_backward_is_finite() -> None:
    predictor = ReferenceBasisCoefficientPredictor(6, 4, 1, 12)
    output = predictor(torch.randn(1, 6), torch.randn(10, 4), torch.rand(10, 1))
    output.coefficients.square().sum().backward()
    gradients = [parameter.grad for parameter in predictor.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_invalid_coefficients_are_rejected() -> None:
    basis = _decomposition().basis
    try:
        basis(torch.tensor([float("nan")]))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN coefficient was accepted")


def test_config_preserves_two_outfit_scope_and_step_caps() -> None:
    assert CONFIG["outfits"] == ["O01", "O08"]
    assert CONFIG["stage_b"]["max_steps"] == 200
    assert CONFIG["stage_c"]["max_steps"] == 496
    assert CONFIG["permissions"]["expand_to_multi_outfit"] is False


def test_future_svd_basis_supports_multiple_outfits_and_projection() -> None:
    teachers = {"A": _teacher(-1.0), "B": _teacher(0.2), "C": _teacher(1.0)}
    decomposition = build_svd_basis(teachers, BOUNDS, ("A", "B", "C"), rank=1)
    assert decomposition.basis.rank == 1
    coefficient = project_residual_onto_basis(decomposition.basis, teachers["B"])
    assert coefficient.shape == (1,) and torch.isfinite(coefficient).all()


def test_future_learned_basis_mode_is_available_but_not_used_here() -> None:
    frozen = _decomposition().basis
    mean, components = frozen.normalized_fields()
    learnable = ExplicitGaussianResidualBasis(mean, components, BOUNDS, trainable=True)
    assert sum(parameter.numel() for parameter in learnable.parameters()) == learnable.explicit_scalar_count
    assert list(_decomposition().basis.parameters()) == []


def test_runner_rng_schema_roundtrip() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    saved = runner.rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    runner.restore_rng(saved)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_stage_b_acceptance_resume_does_not_repeat_optimizer_steps() -> None:
    source = inspect.getsource(runner.run_stage_b)
    acceptance_branch = source.split("if resume_acceptance_only:", 1)[1].split("else:", 1)[0]
    assert "optimizer.step()" not in acceptance_branch
    assert "append_jsonl" not in acceptance_branch
    assert "checkpoint_step_000200.pth" in source
    assert "_restore_stage_b_coefficient_metrics" in source


def test_stage_b_persisted_metrics_can_restore_coefficient_fields() -> None:
    predictor = DiagnosticCoefficientPredictor(4, 1, 8)
    latents = {"O01": torch.tensor([-1.0, 0.0, 0.0, 0.0]), "O08": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    teachers = {"O01": torch.tensor([-1.0]), "O08": torch.tensor([1.0])}
    report = {}
    runner._restore_stage_b_coefficient_metrics(report, predictor, latents, teachers)
    assert set(report) == {
        "coefficients", "teacher_coefficients", "coefficient_mae", "coefficient_separation_ratio",
    }
    assert np.isfinite(report["coefficient_mae"])
    assert np.isfinite(report["coefficient_separation_ratio"])


def test_stage_c_acceptance_resume_does_not_repeat_optimizer_steps() -> None:
    source = inspect.getsource(runner.run_stage_c)
    acceptance_branch = source.split("if resume_acceptance_only:", 1)[1].split("else:", 1)[0]
    assert "optimizer.step()" not in acceptance_branch
    assert "append_jsonl" not in acceptance_branch
    assert "persist=False" in acceptance_branch


def test_checkpoint_parity_replays_reference_model_initialization_rng() -> None:
    source = inspect.getsource(runner.run_stage_c)
    construction = source.split("construction_rng = rng_state()", 1)[1].split("restored_predictor =", 1)[0]
    assert 'restore_rng(context["reference_model_initialization_rng"])' in construction
    assert "construct_reference_model" in construction
    assert "restore_rng(construction_rng)" in construction
