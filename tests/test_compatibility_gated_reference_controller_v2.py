from __future__ import annotations

import inspect
from pathlib import Path

import torch

from scene.compatibility_gated_reference_controller_v2 import (
    PAIR_ORDER,
    CompatibilityEntry,
    CompatibilityGatedReferenceControllerV2,
    CompatibilityPrior,
    ControllerV2Output,
    RoutingThresholds,
    assert_no_forbidden_forward_names,
    construct_tri_mode_runtime,
    controller_v2_forward_argument_names,
    controller_v2_loss_schema,
    route_controller_v2,
    stable_pair_prediction,
)


def _output(
    probabilities: list[float] | None = None,
    *,
    mixedness: float = 0.9,
    valid_count: int = 3,
    pair_weights: list[float] | None = None,
) -> ControllerV2Output:
    probabilities = probabilities or [0.45, 0.35, 0.10, 0.06, 0.04]
    pair_weights = pair_weights or [0.6] * 10
    probs = torch.tensor(probabilities, dtype=torch.float32)
    weights = torch.tensor(pair_weights, dtype=torch.float32)
    return ControllerV2Output(
        reference_feature=torch.zeros(512),
        garment_logits=torch.log(probs),
        garment_probabilities=probs,
        mixedness_logit=torch.logit(torch.tensor(mixedness)),
        mixedness_probability=torch.tensor(mixedness),
        pair_weight_logits=torch.logit(weights),
        all_pair_weights=weights,
        valid_reference_count=valid_count,
    )


def _prior(*, incompatible: set[str] | None = None) -> CompatibilityPrior:
    incompatible = incompatible or set()
    return CompatibilityPrior(
        [
            CompatibilityEntry(
                pair_id=pair,
                compatibility_score=0.0 if pair in incompatible else 1.0,
                compatibility_label=(
                    "INCOMPATIBLE" if pair in incompatible else "COMPATIBLE"
                ),
                manifest_sha256="0" * 64,
                calibration_condition="cond_000017",
            )
            for pair in PAIR_ORDER
        ]
    )


def _thresholds() -> RoutingThresholds:
    return RoutingThresholds(
        mixedness=0.5,
        pair_confidence=0.05,
        provenance="DESIGN_DRY_RUN_ONLY",
    )


def _endpoint_bank() -> dict[str, str]:
    return {
        "O01": "/endpoints/O01.pt",
        "O02": "/endpoints/O02.pt",
        "O03": "/endpoints/O03.pt",
        "O04": "/endpoints/O04.pt",
        "O08": "/endpoints/O08.pt",
    }


def test_v2_has_five_garment_logits() -> None:
    model = CompatibilityGatedReferenceControllerV2(seed=0)
    value = model(torch.zeros(3, 256), torch.ones(3, 1))
    assert value.garment_logits.shape == (5,)
    assert model.head_parameter_counts["garment_head"] == 2565


def test_v2_has_one_mixedness_logit() -> None:
    model = CompatibilityGatedReferenceControllerV2(seed=0)
    value = model(torch.zeros(3, 256), torch.ones(3, 1))
    assert value.mixedness_logit.shape == ()
    assert model.head_parameter_counts["mixedness_head"] == 513


def test_v2_has_ten_pair_weight_logits() -> None:
    model = CompatibilityGatedReferenceControllerV2(seed=0)
    value = model(torch.zeros(3, 256), torch.ones(3, 1))
    assert value.pair_weight_logits.shape == (10,)
    assert value.all_pair_weights.shape == (10,)
    assert model.head_parameter_counts["pair_weight_head"] == 5130
    assert model.parameter_count == 9232


def test_pair_weight_orientation_is_frozen() -> None:
    weights = [0.7] + [0.5] * 9
    value = _output([0.35, 0.45, 0.10, 0.06, 0.04], pair_weights=weights)
    decision = route_controller_v2(value, _prior(), _thresholds())
    assert decision.prediction.predicted_top1 == "O02"
    assert decision.prediction.predicted_pair == "O01_O02"
    assert abs(decision.selected_pair_weight_a - 0.7) < 1e-6
    assert abs(decision.selected_pair_weight_b - 0.3) < 1e-6


def test_gt_pair_not_used_to_select_weight() -> None:
    names = tuple(inspect.signature(route_controller_v2).parameters)
    assert "ground_truth_pair" not in names
    decision = route_controller_v2(_output(), _prior(), _thresholds())
    assert decision.prediction.predicted_pair == "O01_O02"
    assert decision.selected_pair_weight_a == decision.all_pair_weights[0]


def test_mixedness_is_decoupled_from_top2_mass() -> None:
    probabilities = [0.45, 0.35, 0.10, 0.06, 0.04]
    low = route_controller_v2(
        _output(probabilities, mixedness=0.1), _prior(), _thresholds()
    )
    high = route_controller_v2(
        _output(probabilities, mixedness=0.9), _prior(), _thresholds()
    )
    assert low.mode == "SINGLE_ENDPOINT"
    assert low.fallback_reason == "LOW_MIXEDNESS"
    assert high.mode == "DUAL_SUPPORT"


def test_three_modes_are_reachable() -> None:
    single = route_controller_v2(
        _output(mixedness=0.1), _prior(), _thresholds()
    )
    dual = route_controller_v2(_output(), _prior(), _thresholds())
    hard = route_controller_v2(
        _output(), _prior(incompatible={"O01_O02"}), _thresholds()
    )
    assert {single.mode, dual.mode, hard.mode} == {
        "SINGLE_ENDPOINT",
        "DUAL_SUPPORT",
        "HARD_GEOMETRY_SOFT_VA",
    }


def test_single_endpoint_has_one_geometry_support() -> None:
    decision = route_controller_v2(
        _output(mixedness=0.1), _prior(), _thresholds()
    )
    runtime = construct_tri_mode_runtime(decision, _endpoint_bank())
    assert len(runtime.geometry_supports) == 1
    assert len(runtime.visibility_appearance_sources) == 1
    assert runtime.geometry_supports[0].outfit_id == decision.prediction.predicted_top1


def test_dual_support_has_two_immutable_supports() -> None:
    decision = route_controller_v2(_output(), _prior(), _thresholds())
    runtime = construct_tri_mode_runtime(decision, _endpoint_bank())
    assert len(runtime.geometry_supports) == 2
    assert all(
        support.geometry_policy == "IMMUTABLE_ENDPOINT_NO_INTERPOLATION"
        for support in runtime.geometry_supports
    )
    assert not runtime.geometry_interpolation


def test_hard_soft_va_never_interpolates_geometry() -> None:
    decision = route_controller_v2(
        _output(), _prior(incompatible={"O01_O02"}), _thresholds()
    )
    runtime = construct_tri_mode_runtime(decision, _endpoint_bank())
    assert runtime.mode == "HARD_GEOMETRY_SOFT_VA"
    assert len(runtime.geometry_supports) == 1
    assert len(runtime.visibility_appearance_sources) == 2
    assert runtime.geometry_supports[0].outfit_id == decision.dominant_outfit
    assert runtime.geometry_interpolation is False
    assert runtime.geometry_averaging is False
    assert runtime.basis_coefficient_interpolation is False


def test_compatibility_prior_uses_predicted_pair() -> None:
    decision = route_controller_v2(
        _output(), _prior(incompatible={"O01_O02"}), _thresholds()
    )
    assert decision.prediction.predicted_pair == "O01_O02"
    assert decision.compatibility_label == "INCOMPATIBLE"
    assert CompatibilityPrior.lookup_key == "PREDICTED_PAIR"
    assert CompatibilityPrior.query_ground_truth_used is False


def test_no_explicit_pair_blacklist() -> None:
    assert CompatibilityPrior.explicit_pair_blacklist is False
    assert "O01_O03" not in inspect.getsource(
        CompatibilityGatedReferenceControllerV2
    )
    assert "O02_O03" not in inspect.getsource(
        CompatibilityGatedReferenceControllerV2
    )


def test_information_ablation_targets_safe_fallback() -> None:
    dropout = route_controller_v2(
        _output(valid_count=0), _prior(), _thresholds()
    )
    single_reference = route_controller_v2(
        _output(valid_count=1), _prior(), _thresholds()
    )
    assert dropout.mode == single_reference.mode == "SINGLE_ENDPOINT"
    assert (
        dropout.fallback_reason
        == single_reference.fallback_reason
        == "REFERENCE_INFORMATION_INSUFFICIENT"
    )


def test_nuisance_consistency_contract() -> None:
    consistency = controller_v2_loss_schema()["consistency"]
    assert consistency["eligible"] == [
        "mild_blur",
        "mild_mask_erosion",
        "mild_mask_dilation",
        "assignment_permutation",
    ]
    assert "single_reference" in consistency["excluded_information_ablations"]


def test_target_pose_camera_not_in_controller() -> None:
    names = controller_v2_forward_argument_names()
    assert "target_pose" not in names
    assert "target_camera" not in names
    assert_no_forbidden_forward_names(names)


def test_no_gt_id_pair_weight_in_forward() -> None:
    names = controller_v2_forward_argument_names()
    assert names == ("reference_f2", "reference_valid")
    assert not any("ground_truth" in name for name in names)


def test_same_seed_fresh_process_exact() -> None:
    first = CompatibilityGatedReferenceControllerV2(seed=23)
    second = CompatibilityGatedReferenceControllerV2(seed=23)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_cross_seed_initialization_unique() -> None:
    first = CompatibilityGatedReferenceControllerV2(seed=0)
    second = CompatibilityGatedReferenceControllerV2(seed=1)
    assert any(
        not torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_stable_tie_uses_frozen_outfit_order() -> None:
    prediction = stable_pair_prediction(
        torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])
    )
    assert prediction.predicted_top1 == "O01"
    assert prediction.predicted_top2 == "O02"
    assert prediction.predicted_pair == "O01_O02"


def test_no_training_optimizer_or_checkpoint() -> None:
    module = Path(
        inspect.getsourcefile(CompatibilityGatedReferenceControllerV2) or ""
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in module
    assert ".backward(" not in module
    assert "torch.save" not in module
    assert "optimizer.step" not in module
