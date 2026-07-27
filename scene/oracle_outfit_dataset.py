from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from scene.full_dressable_dataset import FullDressableTrainingDataset


class OracleOutfitDataset(Dataset):
    """Outfit-only target-supervision view over FullDressableTrainingDataset."""

    def __init__(self, manifest_path: str | Path, outfit_id: str, split_path: str | Path, split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.outfit_id = outfit_id
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        memberships = [
            name for name, outfit_ids in manifest["splits"].items() if outfit_id in outfit_ids
        ]
        if len(memberships) != 1:
            raise ValueError("outfit_id must belong to exactly one dataset split")
        source = FullDressableTrainingDataset(manifest_path, memberships[0], reference_count=1, seed=0)
        split_document = json.loads(Path(split_path).read_text(encoding="utf-8"))
        if split_document.get("outfit_id") != outfit_id:
            raise ValueError("oracle split outfit_id mismatch")
        wanted = list(split_document["splits"][split])
        records = {}
        for outfit, observation in source.samples:
            if outfit["outfit_id"] == outfit_id:
                records[observation["condition_id"]] = (outfit, observation)
        if set(wanted) != set(records).intersection(wanted):
            missing = sorted(set(wanted).difference(records))
            raise ValueError(f"oracle split conditions are absent from outfit: {missing}")
        self.source = source
        self.records = [records[condition_id] for condition_id in wanted]
        self.condition_ids = wanted
        self.split_fingerprint = split_document["fingerprint"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        outfit, observation = self.records[index]
        target = self.source._observation(outfit, observation, True)
        return {
            "outfit_id": self.outfit_id,
            "condition_id": target["condition_id"],
            "target_rgb": target["rgb"],
            "target_foreground_mask": target["foreground_mask"],
            "target_clothing_mask": target["clothing_mask"],
            "target_pose": target["pose"],
            "target_R_global": target["R_global"],
            "target_Rh": target["R_global"],
            "target_Th": target["Th"],
            "target_K": target["K"],
            "target_w2c": target["w2c"],
            "target_camera": {
                "K": target["K"], "w2c": target["w2c"],
                "width": target["width"], "height": target["height"],
            },
            "oracle_type": "representation_capacity_upper_bound",
        }


class ResumableDeterministicSampler(Sampler[int]):
    def __init__(self, length: int, seed: int, epoch: int = 0, position: int = 0) -> None:
        if length <= 0:
            raise ValueError("sampler length must be positive")
        self.length, self.seed, self.epoch, self.position = int(length), int(seed), int(epoch), int(position)

    def _order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.length, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        while self.position < self.length:
            index = order[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0

    def __len__(self) -> int:
        return self.length - self.position

    def state_dict(self) -> dict[str, int | str]:
        payload = {"length": self.length, "seed": self.seed, "epoch": self.epoch, "position": self.position}
        payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["length"]) != self.length or int(state["seed"]) != self.seed:
            raise ValueError("sampler contract mismatch")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        if self.position < 0 or self.position > self.length:
            raise ValueError("invalid sampler position")
