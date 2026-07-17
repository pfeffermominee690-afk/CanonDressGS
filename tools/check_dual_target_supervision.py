from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.full_dressable_dataset import (  # noqa: E402
    DUAL_TARGET_FIELDS,
    FORBIDDEN_INFERENCE_FIELDS,
    FullDressableInferenceDataset,
    FullDressableTrainingDataset,
    select_forward_conditioning_fields,
)
from utils.rendering_loss_utils import region_aware_dual_target_loss  # noqa: E402


HEIGHT, WIDTH = 6, 8
CONDITIONS = ("cond_a", "cond_b", "cond_c")


def _save_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8), "RGB").save(path)


def _save_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8) * 255, "L").save(path)


def _condition(condition_id: str, index: int) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "source_frame_id": str(index),
        "source_camera_id": str(index),
        "pose": [0.0] * 165,
        "Rh_raw": [0.0, 0.0, 0.0],
        "R_global": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "Th": [0.0, 0.0, 0.0],
        "K": [[4.0, 0.0, 4.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]],
        "w2c": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "c2w": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "width": WIDTH,
        "height": HEIGHT,
        "background": [1.0, 1.0, 1.0],
        "conventions": {},
        "source_checksum": {},
    }


def _build_fixture(root: Path, dual: bool = True, identity_fail: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    foreground = np.ones((HEIGHT, WIDTH), dtype=bool)
    clothing = np.zeros((HEIGHT, WIDTH), dtype=bool)
    clothing[1:5, 2:6] = True
    edit = np.zeros((HEIGHT, WIDTH), dtype=bool)
    edit[1:5, 1:7] = True
    core = np.zeros_like(edit)
    core[2:4, 2:6] = True
    protected = np.zeros_like(edit)
    protected[0, :2] = True
    preserve = ~edit | protected
    transition = edit & ~core & ~protected
    old = clothing.copy()
    revealed = np.zeros_like(edit)
    masks = {
        "foreground_mask": foreground,
        "clothing_mask": clothing,
        "target_edit_mask": edit,
        "target_edit_core_mask": core,
        "target_preserve_mask": preserve,
        "target_transition_mask": transition,
        "target_protected_mask": protected,
        "target_foreground_mask": foreground,
        "target_clothing_mask": clothing,
        "target_old_clothing_mask": old,
        "target_revealed_skin_mask": revealed,
    }
    observations = []
    for index, condition_id in enumerate(CONDITIONS):
        rgb = root / f"{condition_id}_rgb.png"
        edit_rgb = root / f"{condition_id}_edit.png"
        base_rgb = root / f"{condition_id}_base.png"
        _save_rgb(rgb, 40 + index)
        _save_rgb(edit_rgb, 180)
        _save_rgb(base_rgb, 40)
        observation: dict[str, object] = {
            "condition_id": condition_id,
            "rgb": str(rgb),
        }
        for field, array in masks.items():
            path = root / f"{condition_id}_{field}.png"
            _save_mask(path, array)
            if field in {"foreground_mask", "clothing_mask"}:
                observation[field] = str(path)
            elif dual:
                observation[field] = str(path)
        if dual:
            observation.update({
                "target_edit_rgb": str(edit_rgb),
                "target_base_rgb": str(base_rgb),
                "identity_audit_status": "FAIL" if identity_fail and index == 0 else "PASS",
            })
        observations.append(observation)
    manifest = {
        "schema_version": "canondressgs.full_dataset.v1",
        "dataset_kind": "regression",
        "fixture_mode": True,
        "expected_outfits": ["O00"],
        "expected_condition_count": len(CONDITIONS),
        "conditions": [_condition(value, index) for index, value in enumerate(CONDITIONS)],
        "splits": {"train": ["O00"], "val": [], "test": []},
        "outfits": [{"outfit_id": "O00", "metadata": {}, "observations": observations}],
    }
    if dual:
        manifest["supervision_mode"] = "dual_target_region_aware_v1"
    path = root / ("dual.json" if dual else "single.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _dual_loss(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    edit_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return region_aware_dual_target_loss(
        pred_rgb,
        pred_alpha,
        edit_rgb,
        base_rgb,
        masks["edit_core"],
        masks["preserve"],
        masks["protected"],
        masks["transition"],
        masks["clothing"],
        masks["foreground"],
    )


class DualTargetSupervisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="canondressgs_v5_")
        self.root = Path(self.temporary.name)
        self.manifest = _build_fixture(self.root, dual=True)
        self.training = FullDressableTrainingDataset(self.manifest, "train", reference_count=2)
        self.sample = self.training[0]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dual_target_masks_are_valid(self) -> None:
        edit = self.sample["target_edit_mask"].bool()
        preserve = self.sample["target_preserve_mask"].bool()
        protected = self.sample["target_protected_mask"].bool()
        self.assertFalse(torch.any(edit & protected))
        self.assertFalse(torch.any(~(edit | preserve)))
        self.assertTrue(torch.all(self.sample["target_clothing_mask"] <= self.sample["target_foreground_mask"]))

    def test_protected_region_is_preserve_only(self) -> None:
        protected = self.sample["target_protected_mask"].bool()
        self.assertTrue(torch.all(self.sample["target_preserve_mask"].bool()[protected]))
        self.assertFalse(torch.any(self.sample["target_edit_mask"].bool()[protected]))

    def test_dual_target_fields_never_enter_forward(self) -> None:
        forward = select_forward_conditioning_fields(self.sample)
        self.assertFalse(DUAL_TARGET_FIELDS.intersection(forward))
        self.assertFalse({"target_foreground_mask", "target_clothing_mask"}.intersection(forward))

    def test_inference_dataset_excludes_dual_targets(self) -> None:
        sample = FullDressableInferenceDataset(self.manifest, "train", reference_count=2)[0]
        self.assertFalse(FORBIDDEN_INFERENCE_FIELDS.intersection(sample))

    def test_edit_loss_uses_raw_edit_target(self) -> None:
        prediction = torch.ones(1, 3, HEIGHT, WIDTH)
        edit_target = prediction.clone()
        base_target = torch.zeros_like(prediction)
        masks = self._loss_masks()
        losses = _dual_loss(prediction, torch.full((1, 1, HEIGHT, WIDTH), 0.8), edit_target, base_target, masks)
        self.assertEqual(losses["edit"].item(), 0.0)
        self.assertGreater(losses["preserve"].item(), 0.0)

    def test_preserve_loss_uses_base_target(self) -> None:
        prediction = torch.zeros(1, 3, HEIGHT, WIDTH)
        edit_target = torch.ones_like(prediction)
        base_target = prediction.clone()
        masks = self._loss_masks()
        losses = _dual_loss(prediction, torch.full((1, 1, HEIGHT, WIDTH), 0.8), edit_target, base_target, masks)
        self.assertEqual(losses["preserve"].item(), 0.0)
        self.assertGreater(losses["edit"].item(), 0.0)

    def test_region_losses_backward_to_six_heads(self) -> None:
        heads = [torch.tensor(0.1 * (index + 1), requires_grad=True) for index in range(6)]
        combined = sum((index + 1) * head for index, head in enumerate(heads))
        pred_rgb = torch.sigmoid(combined).expand(1, 3, HEIGHT, WIDTH)
        pred_alpha = torch.sigmoid(combined * 0.5).expand(1, 1, HEIGHT, WIDTH)
        masks = self._loss_masks()
        loss = _dual_loss(pred_rgb, pred_alpha, torch.zeros_like(pred_rgb), torch.zeros_like(pred_rgb), masks)["total"]
        loss.backward()
        for head in heads:
            self.assertIsNotNone(head.grad)
            self.assertTrue(torch.isfinite(head.grad))
            self.assertNotEqual(head.grad.item(), 0.0)

    def test_frozen_base_and_backbone_unchanged(self) -> None:
        frozen = torch.nn.Linear(1, 1)
        frozen.requires_grad_(False)
        before = deepcopy(frozen.state_dict())
        trainable = torch.nn.Parameter(torch.tensor(0.2))
        optimizer = torch.optim.SGD([trainable], lr=0.1)
        value = frozen(torch.ones(1, 1)).detach() + trainable
        value.square().backward()
        optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
        for key, value_before in before.items():
            self.assertTrue(torch.equal(value_before, frozen.state_dict()[key]))

    def test_single_target_dataset_regression(self) -> None:
        manifest = _build_fixture(self.root / "single", dual=False)
        sample = FullDressableTrainingDataset(manifest, "train", reference_count=2)[0]
        self.assertIn("target_rgb", sample)
        self.assertFalse(DUAL_TARGET_FIELDS.intersection(sample))
        self.assertEqual(sample["supervision_mode"], "single_target")

    def test_dual_target_checker_fixture(self) -> None:
        report = self.root / "checker.json"
        checker = Path(__file__).resolve().parent / "check_full_dressable_dataset.py"
        result = subprocess.run(
            [sys.executable, str(checker), "--manifest", str(self.manifest), "--regression", "--report", str(report)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["checks"]["dual_target_fields_in_forward"], 0)

    def _loss_masks(self) -> dict[str, torch.Tensor]:
        return {
            "edit_core": self.sample["target_edit_core_mask"].unsqueeze(0),
            "preserve": self.sample["target_preserve_mask"].unsqueeze(0),
            "protected": self.sample["target_protected_mask"].unsqueeze(0),
            "transition": self.sample["target_transition_mask"].unsqueeze(0),
            "clothing": self.sample["target_clothing_mask"].unsqueeze(0),
            "foreground": self.sample["target_foreground_mask"].unsqueeze(0),
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DualTargetSupervisionTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if outcome.wasSuccessful() else "FAIL",
        "tests_run": outcome.testsRun,
        "failures": len(outcome.failures),
        "errors": len(outcome.errors),
    }))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
