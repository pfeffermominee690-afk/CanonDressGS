from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _feature(condition: dict[str, Any], observation: dict[str, Any], root: Path) -> np.ndarray:
    pose = np.asarray(condition["pose"], dtype=np.float64)
    camera = np.asarray(condition["c2w"], dtype=np.float64)
    frame = float(condition.get("source", {}).get("frame", condition.get("frame_id", 0)))
    clothing_path = Path(observation["clothing_mask"])
    foreground_path = Path(observation["foreground_mask"])
    clothing_path = clothing_path if clothing_path.is_absolute() else root / clothing_path
    foreground_path = foreground_path if foreground_path.is_absolute() else root / foreground_path
    with Image.open(clothing_path) as image:
        clothing_area = np.asarray(image.convert("L"), dtype=np.float32).mean() / 255
    with Image.open(foreground_path) as image:
        foreground_area = np.asarray(image.convert("L"), dtype=np.float32).mean() / 255
    vector = np.concatenate((
        pose.reshape(-1),
        camera[:3, :3].reshape(-1),
        camera[:3, 3],
        np.array([frame, clothing_area, foreground_area]),
    ))
    scale = np.std(vector) or 1.0
    return vector / scale


def _farthest_order(features: np.ndarray, ids: list[str], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(ids)))
    selected = [start]
    nearest = np.linalg.norm(features - features[start], axis=1)
    while len(selected) < len(ids):
        nearest[selected] = -1
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        nearest = np.minimum(nearest, np.linalg.norm(features - features[candidate], axis=1))
    return selected


def build_split(manifest_path: Path, outfit_id: str, seed: int, fixture_only: bool) -> dict[str, Any]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    conditions = {item["condition_id"]: item for item in document["conditions"]}
    outfits = [item for item in document["outfits"] if item["outfit_id"] == outfit_id]
    if len(outfits) != 1:
        raise ValueError("outfit_id must resolve uniquely")
    observations = outfits[0]["observations"]
    if not fixture_only and len(observations) != 200:
        raise ValueError("formal oracle split requires exactly 200 conditions")
    if len({item["condition_id"] for item in observations}) != len(observations):
        raise ValueError("duplicate outfit condition IDs")
    ids = [item["condition_id"] for item in observations]
    features = np.stack([_feature(conditions[item["condition_id"]], item, root) for item in observations])
    order = _farthest_order(features, ids, seed)
    if fixture_only:
        train_count = max(1, int(round(len(ids) * 0.7)))
        val_count = max(1, int(round(len(ids) * 0.15))) if len(ids) >= 3 else 0
    else:
        train_count, val_count = 140, 30
    split = {
        "train": [ids[index] for index in order[:train_count]],
        "validation": [ids[index] for index in order[train_count:train_count + val_count]],
        "test": [ids[index] for index in order[train_count + val_count:]],
    }
    fingerprint_payload = {
        "version": "oracle_split_v1", "outfit_id": outfit_id, "seed": seed,
        "fixture_only": fixture_only, "splits": split,
    }
    return {
        **fingerprint_payload,
        "selection": {
            "algorithm": "greedy_farthest_point",
            "features": ["pose", "camera_rotation", "camera_translation", "frame", "clothing_area", "foreground_area"],
            "not_sequential_blocks": True,
        },
        "fingerprint": hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outfit-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    result = build_split(args.manifest.resolve(), args.outfit_id, args.seed, args.fixture_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
