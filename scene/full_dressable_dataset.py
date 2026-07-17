from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DUAL_TARGET_FIELDS = frozenset({
    "target_edit_rgb", "target_base_rgb", "target_edit_mask",
    "target_edit_core_mask", "target_preserve_mask", "target_transition_mask",
    "target_protected_mask", "target_old_clothing_mask",
    "target_revealed_skin_mask",
})

FORWARD_CONDITIONING_FIELDS = frozenset({
    "reference_images", "reference_foreground_masks", "reference_clothing_masks",
    "reference_poses", "reference_R_global", "reference_Rh", "reference_Th",
    "reference_K", "reference_w2c", "reference_cameras", "reference_condition_ids",
    "reference_valid_mask", "target_pose", "target_R_global", "target_Rh",
    "target_Th", "target_K", "target_w2c", "target_camera", "target_condition_id",
})

FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "target_rgb", "target_foreground_mask", "target_clothing_mask",
    "teacher", "anchor_offset_target", "cloth_embedding", "clothing_embedding",
}) | DUAL_TARGET_FIELDS


def select_forward_conditioning_fields(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit reference/pose/camera-only model conditioning boundary."""
    selected = {key: value for key, value in sample.items() if key in FORWARD_CONDITIONING_FIELDS}
    leaked = DUAL_TARGET_FIELDS.intersection(selected)
    if leaked:
        raise RuntimeError(f"dual-target supervision leaked into forward: {sorted(leaked)}")
    return selected


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


class _FullDressableBase(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, reference_count: int = 2,
                 seed: int = 0) -> None:
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "canondressgs.full_dataset.v1":
            raise ValueError("manifest is not CanonDressGS full dataset v1")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if reference_count < 1:
            raise ValueError("reference_count must be positive")
        self.split, self.reference_count, self.seed = split, reference_count, int(seed)
        self.supervision_mode = self.manifest.get("supervision_mode", "single_target")
        if self.supervision_mode not in {"single_target", "dual_target_region_aware_v1"}:
            raise ValueError(f"unknown supervision_mode: {self.supervision_mode}")
        self.conditions = {x["condition_id"]: x for x in self.manifest["conditions"]}
        wanted = set(self.manifest["splits"][split])
        self.outfits = [x for x in self.manifest["outfits"] if x["outfit_id"] in wanted]
        if {x["outfit_id"] for x in self.outfits} != wanted:
            raise ValueError("split names an unknown outfit")
        self.samples = [(outfit, obs) for outfit in self.outfits for obs in outfit["observations"]]

    def __len__(self) -> int:
        return len(self.samples)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def _image(self, value: str, channels: int) -> torch.Tensor:
        mode = "RGB" if channels == 3 else "L"
        with Image.open(self._path(value)) as image:
            array = np.array(image.convert(mode), copy=True)
        tensor = torch.from_numpy(array).float().div_(255)
        return tensor.permute(2, 0, 1).contiguous() if channels == 3 else tensor.unsqueeze(0)

    def _geometry(self, condition_id: str) -> dict[str, Any]:
        item = self.conditions[condition_id]
        return {
            "condition_id": condition_id,
            "pose": _tensor(item["pose"], (165,), "pose"),
            "Rh_raw": _tensor(item["Rh_raw"], (3,), "Rh_raw"),
            "R_global": _tensor(item["R_global"], (3, 3), "R_global"),
            "Th": _tensor(item["Th"], (3,), "Th"),
            "K": _tensor(item["K"], (3, 3), "K"),
            "w2c": _tensor(item["w2c"], (4, 4), "w2c"),
            "c2w": _tensor(item["c2w"], (4, 4), "c2w"),
            "width": int(item["width"]), "height": int(item["height"]),
        }

    def _observation(self, outfit: dict[str, Any], obs: dict[str, Any], images: bool) -> dict[str, Any]:
        value = self._geometry(obs["condition_id"])
        value.update({"outfit_id": outfit["outfit_id"], "outfit_metadata": outfit.get("metadata", {})})
        if images:
            value.update({
                "rgb": self._image(obs["rgb"], 3),
                "foreground_mask": self._image(obs["foreground_mask"], 1),
                "clothing_mask": self._image(obs["clothing_mask"], 1),
            })
        return value

    def _references(self, outfit: dict[str, Any], target_id: str, index: int) -> list[dict[str, Any]]:
        candidates = [x for x in outfit["observations"] if x["condition_id"] != target_id]
        if len(candidates) < self.reference_count:
            raise ValueError("not enough disjoint references")
        return random.Random(self.seed + index * 1000003).sample(candidates, self.reference_count)

    @staticmethod
    def _stack_references(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "reference_images": torch.stack([x["rgb"] for x in items]),
            "reference_foreground_masks": torch.stack([x["foreground_mask"] for x in items]),
            "reference_clothing_masks": torch.stack([x["clothing_mask"] for x in items]),
            "reference_poses": torch.stack([x["pose"] for x in items]),
            "reference_R_global": torch.stack([x["R_global"] for x in items]),
            "reference_Rh": torch.stack([x["R_global"] for x in items]),
            "reference_Th": torch.stack([x["Th"] for x in items]),
            "reference_K": torch.stack([x["K"] for x in items]),
            "reference_w2c": torch.stack([x["w2c"] for x in items]),
            "reference_cameras": [{"K": x["K"], "w2c": x["w2c"], "width": x["width"], "height": x["height"]} for x in items],
            "reference_condition_ids": [x["condition_id"] for x in items],
            "reference_valid_mask": torch.ones(len(items), dtype=torch.float32),
        }


class FullDressableTrainingDataset(_FullDressableBase):
    def __getitem__(self, index: int) -> dict[str, Any]:
        outfit, target_obs = self.samples[index]
        target = self._observation(outfit, target_obs, True)
        references = [self._observation(outfit, x, True) for x in self._references(outfit, target["condition_id"], index)]
        result = self._stack_references(references)
        result.update({
            "target_foreground_mask": target["foreground_mask"],
            "target_clothing_mask": target["clothing_mask"], "target_pose": target["pose"],
            "target_R_global": target["R_global"], "target_Rh": target["R_global"], "target_Th": target["Th"],
            "target_K": target["K"], "target_w2c": target["w2c"],
            "target_camera": {"K": target["K"], "w2c": target["w2c"], "width": target["width"], "height": target["height"]},
            "target_condition_id": target["condition_id"], "outfit_id": outfit["outfit_id"],
            "outfit_metadata": outfit.get("metadata", {}),
            "supervision_mode": self.supervision_mode,
        })
        if self.supervision_mode == "single_target":
            result["target_rgb"] = target["rgb"]
        else:
            required = sorted(DUAL_TARGET_FIELDS)
            missing = [name for name in required if name not in target_obs]
            if missing:
                raise ValueError(f"dual-target observation is missing fields: {missing}")
            result.update({
                "target_edit_rgb": self._image(target_obs["target_edit_rgb"], 3),
                "target_base_rgb": self._image(target_obs["target_base_rgb"], 3),
                **{
                    name: self._image(target_obs[name], 1)
                    for name in required
                    if name not in {"target_edit_rgb", "target_base_rgb"}
                },
            })
        teacher = outfit.get("teacher")
        if teacher is not None:
            result["teacher"] = teacher
        return result


class FullDressableInferenceDataset(_FullDressableBase):
    def __getitem__(self, index: int) -> dict[str, Any]:
        outfit, target_obs = self.samples[index]
        target = self._observation(outfit, target_obs, False)
        refs = [self._observation(outfit, x, True) for x in self._references(outfit, target["condition_id"], index)]
        result = self._stack_references(refs)
        result.update({
            "target_pose": target["pose"], "target_R_global": target["R_global"],
            "target_Rh": target["R_global"], "target_Th": target["Th"], "target_K": target["K"],
            "target_w2c": target["w2c"], "target_camera": {"K": target["K"], "w2c": target["w2c"], "width": target["width"], "height": target["height"]},
            "target_condition_id": target["condition_id"],
        })
        forbidden = FORBIDDEN_INFERENCE_FIELDS.intersection(result)
        if forbidden:
            raise RuntimeError(f"inference boundary violation: {sorted(forbidden)}")
        return result
