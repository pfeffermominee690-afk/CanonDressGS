from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml
from torch import nn

from scene.explicit_gaussian_residual_basis import (
    CHANNEL_TO_BOUND,
    build_svd_basis,
    normalized_residual_dict,
)
from scene.frozen_f2_linear_coefficient_control import FrozenF2ReferenceFeatureExtractor
from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl
from tools import run_explicit_gaussian_residual_basis as legacy
from tools import run_multi_outfit_explicit_basis as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
BOUNDS = CONFIG["basis"]["channel_bounds"]


def _teacher(seed: int, count: int = 9) -> GaussianClothingResiduals:
    generator = torch.Generator().manual_seed(seed)
    fields = {
        "delta_xyz": torch.randn(count, 3, generator=generator) * BOUNDS["xyz"] * .1,
        "delta_log_scaling": torch.randn(count, 3, generator=generator) * BOUNDS["log_scaling"] * .1,
        "delta_rotvec": torch.randn(count, 3, generator=generator) * BOUNDS["rotation"] * .1,
        "delta_opacity_logit": torch.randn(count, 1, generator=generator) * BOUNDS["opacity_logit"] * .1,
        "delta_sh0": torch.randn(count, 1, 3, generator=generator) * BOUNDS["sh0"] * .1,
        "delta_shN": torch.randn(count, 3, 3, generator=generator) * BOUNDS["shN"] * .1,
    }
    return GaussianClothingResiduals(**fields)


def _teachers():
    return {outfit: _teacher(index + 1) for index, outfit in enumerate(runner.TRAIN_OUTFITS)}


def _inputs():
    torch.manual_seed(9)
    images = torch.rand(3, 3, 12, 10)
    masks = torch.zeros(3, 1, 12, 10); masks[:, :, 2:10, 2:8] = 1
    backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.SiLU())
    return images, masks, backbone


def test_multi_outfit_split_is_fixed():
    assert runner.TRAIN_OUTFITS == ("O01", "O02", "O03", "O04", "O08")
    assert runner.HELD_OUT_OUTFIT == "O07" and runner.RESERVE_OUTFIT == "O06"


def test_o07_is_never_used_for_training():
    source = inspect.getsource(runner.run_coefficient_training)
    assert "TRAIN_OUTFITS" in source and "HELD_OUT_OUTFIT" not in source
    assert CONFIG["permissions"]["train_on_o07"] is False


def test_basis_rank_is_at_most_four():
    assert CONFIG["basis"]["candidate_ranks"] == [1, 2, 3, 4]
    assert CONFIG["basis"]["maximum_rank"] == 4
    assert build_svd_basis(_teachers(), BOUNDS, runner.TRAIN_OUTFITS, 4).basis.rank == 4


def test_basis_is_bound_normalized():
    decomposition = build_svd_basis(_teachers(), BOUNDS, runner.TRAIN_OUTFITS, 2)
    mean, components = decomposition.basis.normalized_fields()
    assert set(mean) == set(CHANNELS) and set(components) == set(CHANNELS)
    assert all(CHANNEL_TO_BOUND[name] in BOUNDS for name in CHANNELS)
    assert all(torch.isfinite(value).all() for value in mean.values())


def test_svd_reconstruction_is_deterministic():
    teachers = _teachers()
    first = build_svd_basis(teachers, BOUNDS, runner.TRAIN_OUTFITS, 4)
    second = build_svd_basis(teachers, BOUNDS, runner.TRAIN_OUTFITS, 4)
    assert first.basis.fingerprint() == second.basis.fingerprint()
    assert all(torch.equal(first.teacher_coefficients[key], second.teacher_coefficients[key]) for key in runner.TRAIN_OUTFITS)


def test_coefficient_normalization_uses_train_only():
    source = inspect.getsource(runner.adjudicate_basis_rank)
    assert "TRAIN_OUTFITS" in source and '"held_out_outfit_used": False' in source
    assert "HELD_OUT_OUTFIT" not in source.split("coefficient_matrix =", 1)[1].split("selected_payload", 1)[0]


def test_multi_outfit_predictor_has_no_tanh():
    source = inspect.getsource(MultiOutfitLinearCoefficientControl).lower()
    assert "tanh" not in source and "sigmoid" not in source and "clamp" not in source


def test_multi_outfit_predictor_has_no_outfit_id():
    signature = inspect.signature(MultiOutfitLinearCoefficientControl.forward)
    assert "outfit_id" not in signature.parameters and "cloth_id" not in signature.parameters


def test_balanced_batch_contains_all_five_outfits():
    source = inspect.getsource(runner.run_coefficient_training)
    assert 'batch = [f"{outfit}/{condition}" for outfit in TRAIN_OUTFITS]' in source
    assert CONFIG["coefficient_training"]["balanced_batch_order"] == list(runner.TRAIN_OUTFITS)


def test_target_image_never_enters_prediction_forward():
    parameters = inspect.signature(runner.prediction_feature).parameters
    source = inspect.getsource(runner.prediction_feature)
    assert "target_rgb" not in parameters and "target_image" not in parameters
    assert 'episode["target' not in source


def test_reference_swap_keeps_target_pose_fixed():
    source = inspect.getsource(runner.run_seen_evaluation)
    assert 'sample = context["samples"][key]' in source
    assert 'context["episodes"][f"{swapped_outfit}/{condition}"]' in source
    assert '"target_pose_camera_fixed": True' in source


def test_permutation_invariance():
    images, masks, backbone = _inputs()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 4)
    model = MultiOutfitLinearCoefficientControl(extractor.set_feature_dim, 3)
    order = torch.tensor([2, 0, 1])
    first = model(extractor(images, masks).set_feature).standardized_coefficients
    second = model(extractor(images[order], masks[order]).set_feature).standardized_coefficients
    assert torch.equal(first, second)


def test_basis_is_frozen_during_coefficient_training():
    source = inspect.getsource(runner.run_coefficient_training)
    assert "basis.parameters()" in source and "optimizer = torch.optim.Adam(\n        model.parameters()" in source
    assert "basis.parameters()," not in source


def test_base_backbone_mmlp_renderer_are_frozen():
    images, masks, backbone = _inputs()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 4)
    assert all(not parameter.requires_grad for parameter in extractor.spatial_backbone.parameters())
    assert all(CONFIG["permissions"][key] is False for key in (
        "modify_base", "modify_mmlp", "modify_renderer", "modify_image_backbone"
    ))


def test_historical_outputs_are_immutable(tmp_path: Path):
    historical = tmp_path / "attempt_immutable"; historical.mkdir()
    (historical / "final.json").write_text("{}\n", encoding="utf-8")
    before = legacy.immutable_tree_metadata_fingerprint(historical)
    loaded = runner.load_config(CONFIG_PATH)
    after = legacy.immutable_tree_metadata_fingerprint(historical)
    assert loaded["permissions"]["modify_existing_teachers"] is False
    assert before == after


def test_held_out_reads_frozen_teacher_adjudication_schema():
    current = {"teachers": {"O07": {"status": "PASS"}}}
    legacy_schema = {"outfits": {"O07": {"status": "PASS"}}}
    assert runner._teacher_adjudication_entries(current)["O07"]["status"] == "PASS"
    assert runner._teacher_adjudication_entries(legacy_schema)["O07"]["status"] == "PASS"
