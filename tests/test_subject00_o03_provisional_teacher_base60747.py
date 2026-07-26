from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.second_identity import (
    run_subject00_o03_provisional_teacher_base60747 as runner,
)


def _field() -> SimpleNamespace:
    return SimpleNamespace(
        trainable_support=torch.ones(1, 1, dtype=torch.bool),
        raw_xyz=torch.zeros(1, 3, requires_grad=True),
        raw_log_scaling=torch.zeros(1, 3, requires_grad=True),
        raw_rotvec=torch.zeros(1, 3, requires_grad=True),
        raw_opacity=torch.zeros(1, 1, requires_grad=True),
        raw_sh0=torch.zeros(1, 3, requires_grad=True),
    )


def test_frozen_contract_is_exact() -> None:
    config = runner.load_config(runner.CONFIG_PATH)
    assert config["base"]["step"] == 60747
    assert config["teacher"]["steps"] == 1200
    assert config["teacher"]["seed"] == 20260718
    assert config["teacher"]["loss_weights"] == runner.EXPECTED_LOSS
    assert tuple(config["teacher"]["checkpoint_steps"]) == runner.CHECKPOINT_STEPS
    assert tuple(config["targets"]["slots"]) == runner.SLOTS
    assert tuple(config["targets"]["camera_ids"]) == runner.CAMERAS
    assert (
        config["registration_binding"]["method"]
        == "prediction_only_inverse_grid_sample_from_exact_source_to_target_similarity"
    )
    assert config["registration_binding"]["target_assets_mutated"] is False
    assert config["registration_binding"]["target_resize_pad_or_reencode"] is False
    assert config["paper_eligible"] is False


def test_capacity_loss_uses_target_on_garment_and_base_on_protected() -> None:
    target_rgb = torch.zeros(2, 2, 3)
    target_rgb[0, 0] = 1
    base_rgb = torch.zeros_like(target_rgb)
    prediction = torch.zeros_like(target_rgb, requires_grad=True)
    alpha = torch.zeros(2, 2, 1, requires_grad=True)
    person = torch.ones(2, 2, 1)
    garment = torch.zeros_like(person)
    garment[0, 0] = 1
    protected = person * (1 - garment)
    target = {
        "raw": target_rgb,
        "person": person,
        "garment": garment,
        "protected": protected,
        "boundary": runner.mask_boundary(garment),
    }
    parts = runner.capacity_loss(
        prediction,
        alpha,
        target,
        base_rgb,
        torch.zeros_like(alpha),
        runner.EXPECTED_LOSS,
        _field(),
    )
    assert parts["garment_rgb"] > 0
    assert parts["protected_rgb"] == 0
    parts["total"].backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].abs().sum() > 0


def test_boundary_is_binary_and_nonempty() -> None:
    mask = torch.zeros(7, 7, 1)
    mask[2:5, 2:5] = 1
    boundary = runner.mask_boundary(mask)
    assert boundary.shape == mask.shape
    assert set(torch.unique(boundary).tolist()).issubset({0.0, 1.0})
    assert torch.count_nonzero(boundary) > 0


def test_prediction_registration_warp_is_exact_for_integer_translation() -> None:
    source = torch.zeros(3, 4, 1)
    source[1, 2, 0] = 1
    source_to_target = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]
    )
    target = runner.warp_prediction_to_target(
        source,
        source_to_target,
        target_width=6,
        target_height=6,
        background=0.0,
    )
    assert target.shape == (6, 6, 1)
    assert torch.isclose(target[3, 3, 0], torch.tensor(1.0))
    assert torch.count_nonzero(target > 1e-6) == 1
