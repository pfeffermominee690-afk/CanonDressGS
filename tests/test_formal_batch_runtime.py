from __future__ import annotations

import pytest
import torch

pytest.importorskip("gsplat")

from tools.paper.formal_batch_runtime import (
    BoundedVectorControl,
    LegacyRFFRank4,
    _feature_kind,
    _rank_for,
    _reference_count,
    _training_loss,
    feature_value,
)


def test_method_resolution_matches_frozen_registry() -> None:
    assert _rank_for("A1_Basis_Rank_1") == 1
    assert _rank_for("A1_Basis_Rank_4") == 4
    assert _rank_for("Ours_Seen_Outfit_Explicit_Basis_V1") == 4
    assert _reference_count("A7_Reference_Count_1") == 1
    assert _reference_count("A7_Reference_Count_3") == 3
    assert _feature_kind("B3_Global_Reference_Feature") == "global"
    assert _feature_kind("B4_Clothing_Mean_Only") == "mean"
    assert _feature_kind("B5_Legacy_Complex_Fusion_RF_F") == "rff"


def test_reference_aggregation_is_permutation_invariant() -> None:
    rows = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    valid = torch.ones(3, 1)
    cache = {"episodes": {"O01/cond_000000": {"normal": {
        "f2": rows, "mean": rows[:, :4], "global": rows[:, :4],
        "rff": rows, "valid": valid,
    }}}}
    left = feature_value(cache, "O01/cond_000000", "f2", torch.device("cpu"))
    right = feature_value(
        cache, "O01/cond_000000", "f2", torch.device("cpu"),
        indices=(2, 0, 1),
    )
    assert torch.equal(left, right)


def test_bounded_and_rff_forward_contracts() -> None:
    bounded = BoundedVectorControl(16, 4)
    value = bounded(torch.zeros(1, 16)).standardized_coefficients
    assert value.shape == (4,) and torch.isfinite(value).all()
    rff = LegacyRFFRank4(raw_dim=9, rank=4)
    packed = torch.cat((torch.zeros(3 * 9), torch.ones(3)))
    value = rff(packed).standardized_coefficients
    assert value.shape == (4,) and torch.isfinite(value).all()


def test_frozen_loss_variants_are_finite() -> None:
    prediction = torch.randn(5, 4, requires_grad=True)
    target = torch.randn(5, 4)
    for method in (
        "Ours_Seen_Outfit_Explicit_Basis_V1",
        "A5_Legacy_Endpoint_Supervision",
        "A6_No_Pairwise_Coefficient_Geometry",
    ):
        loss, parts = _training_loss(method, prediction, target)
        assert torch.isfinite(loss)
        assert parts and all(torch.isfinite(value) for value in parts.values())

