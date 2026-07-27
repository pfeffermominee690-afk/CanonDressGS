from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scene.differentiable_support_proxy as proxy  # noqa: E402


RUNNER_PATH = PROJECT_ROOT / "tools/run_differentiable_support_proxy.py"
CONFIG_PATH = PROJECT_ROOT / "configs/research/subject02_differentiable_support_proxy_v1.yaml"


def qualification_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    state_rows: list[dict[str, object]] = []
    pixels: list[dict[str, object]] = []
    values = {"P1": 30.0, "P2": 8.0, "P3": 6.0}
    for outfit in ("O01", "O08"):
        for view in ("front", "back", "left", "right"):
            for state in ("P1", "P2", "P3"):
                value = values[state] + (0.2 if outfit == "O01" else 0.0) + {"front": .3, "back": .2, "left": .1, "right": 0.0}[view]
                state_rows.append({"outfit": outfit, "view": view, "state": state, "proxy_mean": value, "npre_mean": value, "low_support_fraction": {"P1": .1, "P2": .6, "P3": .7}[state], "cloud_required": outfit == "O08" and state == "P3", "cloud_anomaly": True})
                pixels.append({"outfit": outfit, "view": view, "state": state, "pixel_count": 5, "spearman": 1.0})
    return state_rows, pixels


class DifferentiableSupportProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.means = torch.tensor([[4.5, 4.5], [5.0, 4.5]], requires_grad=True)
        self.conic = torch.tensor([[.25, 0., .25], [.25, 0., .25]], requires_grad=True)
        self.opacity = torch.tensor([.5, .5], requires_grad=True)
        self.pixel = torch.tensor([[4.5, 4.5]])

    def test_proxy_uses_additive_support_not_alpha_compositing(self):
        one = proxy.dense_soft_precontributor_count(self.means[:1], self.conic[:1], self.opacity[:1], self.pixel, temperature=.2)
        two = proxy.dense_soft_precontributor_count(self.means[:1].repeat(2, 1), self.conic[:1].repeat(2, 1), self.opacity[:1].repeat(2), self.pixel, temperature=.2)
        self.assertTrue(torch.allclose(two, 2 * one))

    def test_proxy_has_no_transmittance(self):
        source = inspect.getsource(proxy.dense_soft_precontributor_count)
        self.assertNotIn("cumprod", source)
        self.assertNotIn("raster", source)

    def test_proxy_has_no_early_termination(self):
        source = inspect.getsource(proxy.dense_soft_precontributor_count)
        self.assertNotIn("break", source)
        self.assertNotIn("while ", source)

    def test_proxy_a_detaches_covariance(self):
        proxy.dense_soft_precontributor_count(self.means, self.conic, self.opacity, self.pixel, temperature=.2).sum().backward()
        self.assertIsNone(self.conic.grad)

    def test_proxy_a_detaches_opacity(self):
        proxy.dense_soft_precontributor_count(self.means, self.conic, self.opacity, self.pixel, temperature=.2).sum().backward()
        self.assertIsNone(self.opacity.grad)

    def test_proxy_b_ignores_scale_rotation_opacity(self):
        parameters = inspect.signature(proxy.dense_fixed_center_density).parameters
        self.assertFalse({"scale", "rotation", "opacity"}.intersection(parameters))

    def test_proxy_only_backprops_to_xyz(self):
        proxy.dense_soft_precontributor_count(self.means, self.conic, self.opacity, self.pixel, temperature=.2).sum().backward()
        self.assertIsNotNone(self.means.grad)
        self.assertIsNone(self.conic.grad)
        self.assertIsNone(self.opacity.grad)

    def test_proxy_does_not_use_target_in_forward(self):
        for function in (proxy.dense_soft_precontributor_count, proxy.dense_fixed_center_density, proxy.sparse_soft_precontributor_count):
            self.assertFalse(any("target" in name for name in inspect.signature(function).parameters))

    def test_proxy_candidate_grid_is_preregistered(self):
        grid = proxy.build_preregistered_candidate_grid(self.means, width=16, height=16, support_radius_xy=torch.ones_like(self.means) * 4, cell_size=16, construction="unit")
        self.assertEqual(grid.cell_size, 16)
        self.assertFalse(grid.target_fields_used)

    def test_proxy_selection_is_deterministic(self):
        grid = proxy.build_preregistered_candidate_grid(self.means, width=16, height=16, support_radius_xy=torch.ones_like(self.means) * 4, cell_size=16, construction="unit")
        self.assertTrue(torch.equal(proxy.candidates_for_pixels(grid, self.pixel), proxy.candidates_for_pixels(grid, self.pixel)))

    def test_proxy_state_order_matches_npre(self):
        rows, pixels = qualification_fixture()
        result = proxy.qualify_proxy_candidate(rows, pixels)
        self.assertTrue(result["gates"]["A_focus_order_and_ratio"])

    def test_proxy_global_spearman_gate(self):
        rows, pixels = qualification_fixture()
        self.assertGreaterEqual(proxy.qualify_proxy_candidate(rows, pixels)["global_spearman"], .9)

    def test_proxy_pixel_spearman_gate(self):
        rows, pixels = qualification_fixture()
        self.assertTrue(proxy.qualify_proxy_candidate(rows, pixels)["gates"]["D_pixel_spearman"])

    def test_proxy_is_not_saturated(self):
        result = proxy.anti_saturation_metrics(torch.tensor([1., 2., 4., 8., 16.]))
        self.assertGreater(result["p95_p50_ratio"], 1)
        self.assertLess(result["near_max_fraction"], .5)

    def test_proxy_sparse_candidate_matches_dense_toy(self):
        covariance = torch.eye(2).repeat(2, 1, 1) * 4
        radii = proxy.proxy_a_candidate_radii(covariance, self.opacity.detach())
        grid = proxy.build_preregistered_candidate_grid(self.means, width=16, height=16, support_radius_xy=radii, cell_size=16, construction="unit")
        candidates = proxy.candidates_for_pixels(grid, self.pixel)
        dense = proxy.dense_soft_precontributor_count(self.means, self.conic, self.opacity, self.pixel, temperature=.2)
        sparse = proxy.sparse_soft_precontributor_count(self.means, self.conic, self.opacity, self.pixel, candidates, temperature=.2)
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6))

    def test_proxy_empty_region_is_safe(self):
        result = proxy.deterministic_mask_pixels(torch.zeros((4, 4)), 8)
        self.assertEqual(tuple(result.shape), (0, 2))

    def test_proxy_support_grid_is_deterministic(self):
        self.assertTrue(torch.equal(proxy.deterministic_support_grid(17, 13), proxy.deterministic_support_grid(17, 13)))

    def test_gradient_direction_gate(self):
        means = torch.tensor([[0., 0.]], requires_grad=True)
        pixel = torch.tensor([[7., 0.]])
        conic = torch.tensor([[.25, 0., .25]])
        opacity = torch.tensor([.5])
        loss = -proxy.dense_soft_precontributor_count(means, conic, opacity, pixel, temperature=.2).mean()
        loss.backward()
        self.assertGreater(float(-means.grad[0, 0]), 0)

    def test_protected_gaussians_have_zero_gradient(self):
        full = torch.tensor([[4.5, 4.5], [5., 4.5], [7., 7.]], requires_grad=True)
        editable = full[torch.tensor([0, 1])]
        proxy.dense_fixed_center_density(editable, self.pixel, radius_pixels=4.).sum().backward()
        self.assertTrue(torch.equal(full.grad[2], torch.zeros(2)))

    def test_no_optimizer_is_created(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("Optimizer(", source)

    def test_editable_pool_is_unchanged(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["editable_pool"]["count"], 169106)
        self.assertEqual(config["editable_pool"]["indices_sha256"], "232124458848a684cd98bef1b162ec889f1e3e4c80c0ef1a5caebcceb4edb011")

    def test_previous_outputs_are_unchanged(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("rmtree", source)
        self.assertNotIn("unlink(", source)

    def test_frozen_branches_are_unchanged(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["frozen_branches"]["research/screen-space-placement-objective-20260719"], "6bfd311a1f558db96c505e03bea24f6877c6ea43")


if __name__ == "__main__":
    unittest.main(verbosity=2)
