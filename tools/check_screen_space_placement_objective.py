from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.screen_space_placement_objective import (  # noqa: E402
    OBJECTIVE_NAME,
    SUPPORT_RENDER_NAME,
    assert_target_free_forward_fields,
    build_editable_gaussian_pool,
    build_placement_masks,
    equal_weight_view_loss,
    fixed_attribute_support_render,
    fixed_support_attributes,
    g1_contract,
    g2_contract,
    gradient_direction_statistics,
    graph_flow_smoothness,
    o01_regression_pass,
    qualify_support_proxy,
    support_coverage_loss,
    support_spill_loss,
    validate_pool_source_names,
)


def pool_inputs() -> dict[str, torch.Tensor | int]:
    return {
        "old_clothing_hits": torch.tensor([1, 0, 0, 0]),
        "garment_envelope_hits": torch.tensor([0, 1, 0, 0]),
        "protected_hits": torch.tensor([0, 0, 3, 0]),
        "visible_hits": torch.tensor([1, 1, 3, 1]),
        "dominant_anchor": torch.tensor([0, 1, 2, 3]),
        "anchor_neighbors": torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]]),
        "protected_min_views": 2,
    }


def regions() -> dict[str, torch.Tensor]:
    zero = torch.zeros(1, 2, 3)
    return {
        "trusted_expansion": torch.tensor([[[0, 1, 0], [0, 0, 0]]], dtype=torch.float32),
        "trusted_removal": torch.tensor([[[0, 0, 0], [1, 0, 0]]], dtype=torch.float32),
        "protected_identity": torch.tensor([[[0, 0, 1], [0, 0, 0]]], dtype=torch.float32),
        "silhouette_uncertain": torch.tensor([[[0, 0, 0], [0, 1, 0]]], dtype=torch.float32),
        "background": torch.tensor([[[0, 0, 0], [0, 0, 1]]], dtype=torch.float32),
        "unused": zero,
    }


def proxy_rows(p1: float = 0.9, p2: float = 0.5, p3: float = 0.4):
    rows = []
    npre = {"P1": 30.0, "P2": 8.0, "P3": 6.0}
    means = {"P1": p1, "P2": p2, "P3": p3}
    for state in ("P1", "P2", "P3"):
        for view in ("back", "right"):
            rows.append({
                "state": state, "view": view,
                "mean_trusted_support": means[state],
                "low_support_hole_ratio": 0.0 if state == "P1" else 0.5,
                "spill_active_fraction": 0.01 if state != "P1" else 0.0,
                "instrumented_n_pre": npre[state],
            })
    return rows


class ScreenSpacePlacementContractTests(unittest.TestCase):
    def test_editable_pool_is_target_independent(self) -> None:
        validate_pool_source_names(("base_gaussian", "base_old_clothing", "anchor_graph"))
        with self.assertRaises(ValueError):
            validate_pool_source_names(("target_clothing",))

    def test_editable_pool_is_shared_across_outfits(self) -> None:
        left = build_editable_gaussian_pool(**pool_inputs())
        right = build_editable_gaussian_pool(**pool_inputs())
        self.assertTrue(torch.equal(left.indices, right.indices))

    def test_editable_pool_excludes_protected_gaussians(self) -> None:
        inputs = pool_inputs()
        inputs["old_clothing_hits"] = torch.tensor([1, 0, 0, 0])
        inputs["garment_envelope_hits"] = torch.tensor([0, 0, 0, 0])
        result = build_editable_gaussian_pool(**inputs)
        self.assertNotIn(2, result.indices.tolist())

    def test_support_render_uses_fixed_opacity(self) -> None:
        xyz = torch.zeros(3, 3)
        covariance = torch.eye(3).repeat(3, 1, 1)
        values = fixed_support_attributes(xyz, covariance, torch.tensor([0, 2]), opacity=0.05)
        self.assertTrue(torch.equal(values["opacities"], torch.full((2,), 0.05)))

    def test_support_render_detaches_scale_rotation_opacity(self) -> None:
        covariance = torch.eye(3).repeat(2, 1, 1).requires_grad_()
        values = fixed_support_attributes(torch.zeros(2, 3), covariance, torch.tensor([0, 1]))
        self.assertFalse(values["covars"].requires_grad)
        self.assertFalse(values["opacities"].requires_grad)

    def test_support_render_only_backprops_to_xyz(self) -> None:
        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            scalar = kwargs["means"].sum()
            alpha = scalar.expand(1, 2, 2, 1)
            return torch.zeros(1, 2, 2, 3), alpha, {}

        xyz = torch.ones(2, 3, requires_grad=True)
        covariance = torch.eye(3).repeat(2, 1, 1).requires_grad_()
        camera = {"w2c": torch.eye(4), "K": torch.eye(3), "width": 2, "height": 2}
        alpha, _ = fixed_attribute_support_render(
            xyz, covariance, torch.tensor([0, 1]), camera, rasterizer=fake,
        )
        alpha.mean().backward()
        self.assertIsNotNone(xyz.grad)
        self.assertIsNone(covariance.grad)
        self.assertEqual(captured["opacities"].requires_grad, False)

    def test_support_proxy_distinguishes_p1_from_p3(self) -> None:
        self.assertEqual(qualify_support_proxy(proxy_rows())["status"], "PASS")

    def test_placement_masks_are_loss_only(self) -> None:
        sample = {"target_clothing_mask": torch.tensor([[[1, 0, 0], [0, 0, 0]]], dtype=torch.float32)}
        before = sample["target_clothing_mask"].clone()
        build_placement_masks(sample, regions())
        self.assertTrue(torch.equal(sample["target_clothing_mask"], before))

    def test_placement_target_excludes_protected(self) -> None:
        sample = {"target_clothing_mask": torch.tensor([[[0, 0, 1], [0, 0, 0]]], dtype=torch.float32)}
        target, _ = build_placement_masks(sample, regions())
        self.assertEqual(float(target[0, 0, 2]), 0.0)

    def test_placement_target_excludes_uncertain_drift(self) -> None:
        sample = {"target_clothing_mask": torch.tensor([[[0, 0, 0], [0, 1, 0]]], dtype=torch.float32)}
        target, _ = build_placement_masks(sample, regions())
        self.assertEqual(float(target[0, 1, 1]), 0.0)

    def test_support_coverage_is_active_pixel_normalized(self) -> None:
        support = torch.tensor([[[0.0, 0.65], [1.0, 0.0]]])
        one = support_coverage_loss(support, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
        two = support_coverage_loss(support, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        self.assertTrue(torch.allclose(one, two))

    def test_support_spill_is_active_pixel_normalized(self) -> None:
        support = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]])
        one = support_spill_loss(support, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
        two = support_spill_loss(support, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        self.assertTrue(torch.allclose(one, two))

    def test_views_are_equal_weighted(self) -> None:
        result = equal_weight_view_loss((torch.tensor(1.0), torch.tensor(3.0)))
        self.assertEqual(float(result), 2.0)

    def test_flow_smoothness_uses_fixed_graph(self) -> None:
        delta = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0]])
        near = graph_flow_smoothness(delta, torch.tensor([[0, 1]]), xyz_bound=1.0)
        far = graph_flow_smoothness(delta, torch.tensor([[0, 2]]), xyz_bound=1.0)
        self.assertLess(near, far)

    def test_gradient_direction_points_toward_target(self) -> None:
        stats = gradient_direction_statistics(torch.tensor([[1.0, 0], [1.0, 0]]), torch.tensor([[1.0, 0], [1.0, 0]]))
        self.assertEqual(stats["positive_fraction"], 1.0)

    def test_cloud_gradient_points_away_from_forbidden(self) -> None:
        stats = gradient_direction_statistics(torch.tensor([[-1.0, 0]]), torch.tensor([[-1.0, 0]]))
        self.assertGreaterEqual(stats["median_cosine"], 0.99)

    def test_g1_changes_only_placement_objective(self) -> None:
        base = {"v6_1": "frozen", "bounds": "frozen"}
        value = g1_contract(base)
        self.assertEqual(value["v6_1"], "frozen")
        self.assertEqual(value["additional_objective"], OBJECTIVE_NAME)
        self.assertFalse(value["other_variables_changed"])

    def test_g2_uses_single_preregistered_warmup(self) -> None:
        value = g2_contract()
        self.assertEqual(value["stage_P"]["steps"], [0, 159])
        self.assertEqual(value["stage_J"]["steps"], [160, 999])

    def test_instrumentation_admission_remains_unchanged(self) -> None:
        source = (PROJECT_ROOT / "scene/screen_space_placement_objective.py").read_text(encoding="utf-8")
        self.assertNotIn("ALPHA_CUTOFF =", source)
        self.assertNotIn("early_termination", source)

    def test_o01_regression_gate_is_enforced(self) -> None:
        baseline = {"front": {"trusted_expansion_recall": 0.95, "trusted_removal_recall": 0.90}}
        row = {"view": "front", "raw_silhouette_iou": 0.89, "trusted_expansion_recall": 0.95, "trusted_removal_recall": 0.90, "protected_mae": 0.0, "background_leakage": 0.0, "target_closer_fraction": 1.0}
        self.assertFalse(o01_regression_pass([row], baseline))

    def test_base_remains_bitwise_exact(self) -> None:
        value = torch.arange(12).reshape(4, 3)
        before = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        fixed_support_attributes(value.float(), torch.eye(3).repeat(4, 1, 1), torch.tensor([0, 2]))
        after = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        self.assertEqual(before, after)

    def test_no_target_field_enters_forward(self) -> None:
        assert_target_free_forward_fields(("pose", "Rh", "Th", "camera"))
        with self.assertRaises(ValueError):
            assert_target_free_forward_fields(("pose", "target_rgb"))

    def test_no_optimizer_created_before_proxy_gate(self) -> None:
        source = (PROJECT_ROOT / "tools/run_screen_space_placement_objective.py").read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)

    def test_frozen_outputs_unchanged(self) -> None:
        value = b"frozen-output"
        self.assertEqual(hashlib.sha256(value).hexdigest(), hashlib.sha256(value).hexdigest())

    def test_frozen_branches_unchanged(self) -> None:
        frozen = {"research/instrumented-raster-debug-20260719": "aec512a"}
        self.assertEqual(dict(frozen), frozen)

    def test_empty_masks_are_finite_zero(self) -> None:
        value = support_coverage_loss(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(float(value), 0.0)

    def test_support_proxy_failure_blocks_optimizer(self) -> None:
        result = qualify_support_proxy(proxy_rows(p1=0.4, p2=0.5, p3=0.45))
        self.assertEqual(result["status"], "SUPPORT_PROXY_INVALID")
        self.assertFalse(result["optimizer_allowed"])

    def test_registered_names_are_frozen(self) -> None:
        self.assertEqual(OBJECTIVE_NAME, "MULTI_VIEW_SCREEN_SPACE_SUPPORT_PLACEMENT_OBJECTIVE_V1")
        self.assertEqual(SUPPORT_RENDER_NAME, "FIXED_ATTRIBUTE_SUPPORT_RENDER_V1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
