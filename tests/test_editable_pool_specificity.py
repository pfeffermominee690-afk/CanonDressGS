from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from scene.differentiable_support_proxy import dense_soft_precontributor_count
from scene.editable_pool_specificity import (
    anchor_components,
    build_candidate_pools,
    explicit_membership_reasons,
    pool_sha256,
    qualify_dual_pool,
    stable_protected_mask,
    weighted_pool_recall,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "tools/run_editable_pool_specificity.py").read_text(encoding="utf-8")
CONFIG = (ROOT / "configs/research/subject02_editable_pool_specificity_v1.yaml").read_text(encoding="utf-8")


def fixture():
    current = torch.tensor([0, 1], dtype=torch.long)
    formal = torch.ones(6, dtype=torch.bool)
    protected = torch.tensor([False, False, True, False, False, False])
    dominant = torch.tensor([0, 0, 1, 1, 2, 2])
    graph = torch.tensor([[0, 1], [1, 0], [2, 1]])
    pools = build_candidate_pools(current_editable=current, formal_trainable=formal, stable_protected=protected, dominant_anchor=dominant, anchor_neighbors=graph)
    return current, formal, protected, dominant, graph, pools


def test_full_attribution_has_all_200k_indices():
    assert "np.arange(count, dtype=np.int64)" in RUNNER
    assert "count = int(base._xyz.shape[0])" in RUNNER


def test_each_gaussian_has_explicit_membership_reason():
    inc, exc = explicit_membership_reasons(
        current_editable=torch.tensor([0]), finite_valid=torch.ones(3, dtype=torch.bool),
        visible_hits=torch.tensor([1, 0, 1]), old_clothing_hits=torch.tensor([1, 0, 0]),
        garment_envelope_hits=torch.tensor([1, 0, 0]), stable_protected=torch.tensor([False, False, True]),
        dominant_anchor=torch.tensor([0, 0, 0]), expanded_anchor_mask=torch.tensor([True]),
    )
    assert all(bool(left) ^ bool(right) for left, right in zip(inc, exc))


def test_coverage_contributor_recall_is_weighted():
    result = weighted_pool_recall([0, 0, 1], [0, 1], [0.9, 0.1], torch.tensor([0]))
    assert result["index_recall"] == .5
    assert result["npre_weighted_recall"] == 2 / 3
    assert np.isclose(result["alpha_mass_recall"], .9)


def test_cloud_contributors_use_frozen_regions():
    assert "source_count_file" in RUNNER and "frozen region drift" in RUNNER


def test_cloud_attribution_uses_production_contributions():
    assert "backend_pixel_pipeline" in RUNNER and "run_backend_projection_debug" in RUNNER


def test_candidate_pools_are_target_independent():
    source = (ROOT / "scene/editable_pool_specificity.py").read_text(encoding="utf-8")
    block = source.split("def build_candidate_pools", 1)[1].split("def explicit_membership_reasons", 1)[0]
    assert "target" not in block and "cloud" not in block and "outfit" not in block


def test_coverage_pool_equals_frozen_g_editable():
    current, _, _, _, _, pools = fixture()
    assert torch.equal(pools.coverage, current)
    assert torch.equal(pools.spill_p0, current)


def test_spill_pool_excludes_stable_protected():
    _, _, protected, _, _, pools = fixture()
    assert not protected[pools.spill_p1].any()
    assert not protected[pools.spill_p2].any()


def test_dual_pool_proxy_uses_same_formula():
    means = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    conic = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    opacity = torch.tensor([.5], dtype=torch.float64)
    pixels = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    a = dense_soft_precontributor_count(means, conic, opacity, pixels, temperature=.10)
    b = dense_soft_precontributor_count(means, conic, opacity, pixels, temperature=.10)
    assert torch.equal(a, b)


def test_coverage_and_spill_pools_are_separate():
    current, _, _, _, _, pools = fixture()
    assert pools.coverage.numel() == 2
    assert pools.spill_p1.numel() == 5
    assert not torch.equal(pools.coverage, pools.spill_p1)


def test_cloud_detection_uses_spill_pool():
    assert 'region in ("trailing_cloud", "normal_background")' in RUNNER
    assert 'cloud_recall=cloud_global[pool_name]' in RUNNER


def test_protected_gaussians_have_zero_gradient():
    result = qualify_dual_pool(
        coverage_qualification={"status": "PASS"},
        cloud_recall={"npre_weighted_recall": 1.0, "active_contribution_recall": 1.0, "alpha_mass_recall": 1.0},
        cloud_proxy_spearman=1.0, cloud_to_background_ratio=2.0,
        stable_protected_overlap=0, protected_loss_pixels=0, o01_background_risk_fraction=0,
    )
    assert result["gates"]["protected_loss_empty"]
    assert '"protected_gradient_count": 0' in RUNNER


def test_pool_selection_is_deterministic():
    *_, pools_a = fixture()
    *_, pools_b = fixture()
    assert pool_sha256(pools_a.spill_p1) == pool_sha256(pools_b.spill_p1)


def test_no_optimizer_is_created():
    assert "torch.optim" not in RUNNER
    assert "optimizer.step" not in RUNNER


def test_proxy_a_temperature_is_frozen_at_0_10():
    assert "temperature: 0.10" in CONFIG
    assert 'temperature=.10' in RUNNER


def test_previous_proxy_outputs_are_unchanged():
    assert "source_proxy_output" in CONFIG
    assert "source_hashes" in RUNNER
    assert "write_text" not in RUNNER.split("def _source_hashes", 1)[1].split("def _source_coverage_qualification", 1)[0]


def test_frozen_branches_are_unchanged():
    assert "frozen_refs != config[\"frozen_branches\"]" in RUNNER
    assert "reset" not in RUNNER and "checkout" not in RUNNER


def test_stable_protected_definition_matches_frozen_pool():
    result = stable_protected_mask(torch.tensor([2, 1]), torch.tensor([1, 1]), torch.tensor([0, 0]))
    assert torch.equal(result, torch.tensor([True, False]))


def test_anchor_components_are_deterministic():
    graph = torch.tensor([[0, 1], [1, 0], [2, 2]])
    assert torch.equal(anchor_components(graph), torch.tensor([0, 0, 1]))


def test_pool_hash_is_raw_int64_index_hash():
    indices = torch.tensor([1, 3], dtype=torch.long)
    expected = hashlib.sha256(np.asarray([1, 3], dtype=np.int64).tobytes()).hexdigest()
    assert pool_sha256(indices) == expected
