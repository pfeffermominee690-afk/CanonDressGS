from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.clothing_loss_utils import (
    anchor_graph_smoothness_loss,
    anchor_offset_supervision_loss,
    masked_smooth_l1_loss,
    non_clothing_region_loss,
    offset_magnitude_regularization,
)


def main() -> None:
    torch.manual_seed(0)

    target = torch.randn(4, 3)
    prediction = target.clone().requires_grad_(True)
    exact_loss = masked_smooth_l1_loss(prediction, target)
    if not torch.equal(exact_loss, torch.zeros_like(exact_loss)):
        raise AssertionError("identical prediction and target did not produce zero loss")
    print("masked smooth l1 test: PASS")

    prediction_masked = torch.tensor([[0.0, 0.0], [4.0, -3.0]])
    target_masked = torch.zeros_like(prediction_masked)
    mask_1d = torch.tensor([1.0, 0.0])
    mask_2d = mask_1d[:, None]
    loss_1d = masked_smooth_l1_loss(prediction_masked, target_masked, valid_mask=mask_1d)
    loss_2d = masked_smooth_l1_loss(prediction_masked, target_masked, valid_mask=mask_2d)
    if not torch.equal(loss_1d, torch.zeros_like(loss_1d)) or not torch.equal(
        loss_2d, torch.zeros_like(loss_2d)
    ):
        raise AssertionError("broadcast mask did not suppress invalid rows")
    print("mask broadcasting test: PASS")

    weighted_prediction = torch.tensor([[1.0], [3.0]])
    weighted_target = torch.zeros_like(weighted_prediction)
    unweighted = masked_smooth_l1_loss(weighted_prediction, weighted_target)
    weighted = masked_smooth_l1_loss(
        weighted_prediction,
        weighted_target,
        weight=torch.tensor([1.0, 0.0]),
    )
    if not weighted < unweighted or not torch.allclose(weighted, torch.tensor(0.5)):
        raise AssertionError("element weights did not change the weighted mean correctly")
    print("weighted loss test: PASS")

    zero_mask_loss = masked_smooth_l1_loss(
        torch.ones(3, 2),
        torch.zeros(3, 2),
        valid_mask=torch.zeros(3, 1),
    )
    if not torch.isfinite(zero_mask_loss) or zero_mask_loss.item() != 0:
        raise AssertionError("all-zero mask did not return a finite zero")
    print("zero mask stability test: PASS")

    predicted_offsets = {
        "delta_xyz": torch.ones(4, 3),
        "delta_scaling": torch.ones(4, 3),
        "delta_opacity": torch.ones(4, 1),
    }
    partial = anchor_offset_supervision_loss(
        predicted_offsets,
        {"delta_xyz": torch.zeros(4, 3)},
    )
    if partial["xyz"] <= 0 or partial["scaling"].item() != 0 or partial["opacity"].item() != 0:
        raise AssertionError("partial target fields were not handled correctly")
    if not torch.equal(partial["total"], partial["xyz"]):
        raise AssertionError("partial supervision total is incorrect")
    print("partial target test: PASS")

    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    constant_offsets = torch.ones(3, 3)
    discontinuous_offsets = constant_offsets.clone()
    discontinuous_offsets[1, 0] = 4.0
    constant_smoothness = anchor_graph_smoothness_loss(constant_offsets, edges)
    discontinuous_smoothness = anchor_graph_smoothness_loss(discontinuous_offsets, edges)
    empty_smoothness = anchor_graph_smoothness_loss(
        constant_offsets,
        torch.empty(0, 2, dtype=torch.long),
    )
    if constant_smoothness.item() != 0 or discontinuous_smoothness <= 0:
        raise AssertionError("anchor graph smoothness does not detect discontinuities")
    if not torch.isfinite(empty_smoothness) or empty_smoothness.item() != 0:
        raise AssertionError("empty edge graph did not return finite zero")
    print("anchor smoothness test: PASS")

    region_offsets = {
        "delta_xyz": torch.tensor([[3.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        "delta_scaling": torch.zeros(2, 3),
        "delta_opacity": torch.zeros(2, 1),
    }
    region = torch.tensor([[1.0], [0.0]])
    noncloth = non_clothing_region_loss(region_offsets, region)
    clothing_only_offsets = {key: value.clone() for key, value in region_offsets.items()}
    clothing_only_offsets["delta_xyz"][1] = 0
    clothing_only = non_clothing_region_loss(clothing_only_offsets, region)
    if noncloth["xyz"] <= 0 or clothing_only["total"].item() != 0:
        raise AssertionError("non-clothing loss penalized the wrong region")
    print("non-clothing region test: PASS")

    magnitude = offset_magnitude_regularization(predicted_offsets)
    if set(magnitude) != {"xyz", "scaling", "opacity", "total"}:
        raise AssertionError("magnitude regularization returned unexpected keys")
    if any(value.ndim != 0 for value in magnitude.values()):
        raise AssertionError("magnitude regularization values are not scalar tensors")
    if not torch.allclose(
        magnitude["total"],
        magnitude["xyz"] + magnitude["scaling"] + magnitude["opacity"],
    ):
        raise AssertionError("magnitude regularization total is incorrect")
    print("offset magnitude test: PASS")

    gradient_prediction = torch.randn(5, 3, requires_grad=True)
    gradient_loss = masked_smooth_l1_loss(
        gradient_prediction,
        torch.zeros_like(gradient_prediction),
        valid_mask=torch.ones(5, 1),
        weight=torch.linspace(0.2, 1.0, 5),
    )
    gradient_loss.backward()
    if gradient_prediction.grad is None or not torch.isfinite(gradient_prediction.grad).all():
        raise AssertionError("loss backward did not produce finite prediction gradients")
    print("loss gradient test: PASS")

    invalid_checks = [
        lambda: masked_smooth_l1_loss(torch.zeros(2, 3), torch.zeros(2, 2)),
        lambda: masked_smooth_l1_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            valid_mask=torch.ones(4),
        ),
        lambda: anchor_graph_smoothness_loss(
            torch.zeros(3, 2),
            torch.tensor([[0.0, 1.0]]),
        ),
        lambda: non_clothing_region_loss(
            predicted_offsets,
            torch.full((4, 1), 1.5),
        ),
    ]
    for check in invalid_checks:
        try:
            check()
        except (TypeError, ValueError, IndexError):
            continue
        raise AssertionError("invalid loss input did not raise a clear exception")
    print("invalid loss input test: PASS")


if __name__ == "__main__":
    main()
