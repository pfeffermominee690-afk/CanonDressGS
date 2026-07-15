from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.dressable_dataset import (
    ImageConditionedEpisodeDataset,
    image_conditioned_episode_collate,
)
from tools.build_synthetic_image_conditioned_manifest import (
    build_synthetic_image_conditioned_manifest,
)
from utils.dressable_camera_utils import validate_camera_data


def _write_manifest(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _all_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_tensors(item)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="canondress_episode_dataset_") as temp_dir:
        root = Path(temp_dir)
        manifest_path = build_synthetic_image_conditioned_manifest(root)
        original_manifest_bytes = manifest_path.read_bytes()
        dataset = ImageConditionedEpisodeDataset(
            manifest_path,
            split="train",
            reference_count=4,
            seed=17,
        )
        if len(dataset) != 12 or dataset.identity_id != "synthetic_subject":
            raise AssertionError("manifest was not parsed into expected episodes")
        print("manifest parse test: PASS")

        if {dataset[index]["cloth_name"] for index in range(len(dataset))} != {
            "cloth_red",
            "cloth_blue",
        }:
            raise AssertionError("train dataset contains validation/test clothing")
        print("split isolation test: PASS")
        val_dataset = ImageConditionedEpisodeDataset(manifest_path, "val", 3, seed=17)
        if dataset.get_cloth_id_map() != val_dataset.get_cloth_id_map() or dataset.get_cloth_id_map() != {
            "cloth_blue": 0,
            "cloth_green": 1,
            "cloth_red": 2,
            "cloth_yellow": 3,
        }:
            raise AssertionError("cloth ID map is not globally stable")
        print("stable cloth map test: PASS")

        episode = dataset[0]
        target = (episode["target_frame_id"], episode["target_view_id"])
        references = set(zip(episode["reference_frame_ids"], episode["reference_view_ids"]))
        if target in references:
            raise AssertionError("target observation leaked into reference set")
        print("reference target separation test: PASS")
        if episode["reference_images"].shape[0] != 4:
            raise AssertionError("fixed reference count is not K=4")
        print("fixed reference count test: PASS")

        variable = ImageConditionedEpisodeDataset(manifest_path, "train", (2, 5), seed=3)
        observed_counts = {variable[0]["reference_images"].shape[0] for _ in range(20)}
        if not observed_counts.issubset({2, 3, 4, 5}) or len(observed_counts) < 2:
            raise AssertionError("variable reference sampling did not vary within bounds")
        print("variable reference count test: PASS")
        val_again = ImageConditionedEpisodeDataset(manifest_path, "val", 3, seed=17)
        first = val_dataset[0]
        second = val_again[0]
        if first["reference_frame_ids"] != second["reference_frame_ids"]:
            raise AssertionError("validation reference sampling is not deterministic")
        print("deterministic validation test: PASS")
        if not all(path.is_absolute() for path in (
            dataset._resolved_clothes["cloth_red"]["frames"][0]["rgb"],
            dataset._resolved_clothes["cloth_red"]["frames"][0]["camera"],
        )):
            raise AssertionError("relative manifest paths were not resolved")
        print("path resolution test: PASS")

        if episode["reference_images"].shape != (4, 3, 64, 64) or torch.any(
            episode["reference_images"] < 0
        ) or torch.any(episode["reference_images"] > 1):
            raise AssertionError("RGB tensor shape/range is invalid")
        print("rgb shape range test: PASS")
        masks = (
            episode["reference_cloth_masks"],
            episode["reference_foreground_masks"],
            episode["target_foreground_mask"],
            episode["target_clothing_mask"],
        )
        if any(mask.dtype != torch.float32 or torch.any(mask < 0) or torch.any(mask > 1) for mask in masks):
            raise AssertionError("mask tensor shape/range is invalid")
        print("mask shape range test: PASS")
        fallback_episode = dataset.sample_episode(5, deterministic=True)
        if not fallback_episode["metadata"]["target_clothing_mask_fallback"] or not torch.equal(
            fallback_episode["target_clothing_mask"],
            fallback_episode["target_foreground_mask"],
        ):
            raise AssertionError("missing clothing mask did not fall back to foreground")
        print("clothing mask fallback test: PASS")
        if episode["reference_poses"].shape != (4, 6) or not torch.isfinite(
            episode["reference_poses"]
        ).all():
            raise AssertionError("pose shape/finite validation failed")
        print("pose shape finite test: PASS")
        for camera in (*episode["reference_cameras"], episode["target_camera"]):
            validate_camera_data(camera)
        print("camera validation test: PASS")
        target_data = episode["anchor_offset_target"]
        if target_data is None or target_data["anchor_xyz"].shape != (8, 3):
            raise AssertionError("anchor teacher target was not loaded")
        print("anchor target loading test: PASS")
        if image_conditioned_episode_collate([episode]) is not episode:
            raise AssertionError("batch-size-one collate changed the episode")
        try:
            image_conditioned_episode_collate([episode, episode])
        except ValueError:
            pass
        else:
            raise AssertionError("collate accepted batch_size > 1")
        print("episode collate test: PASS")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        insufficient = copy.deepcopy(manifest)
        insufficient["clothes"]["cloth_red"]["frames"] = insufficient["clothes"]["cloth_red"]["frames"][:2]
        try:
            ImageConditionedEpisodeDataset(
                _write_manifest(root, "insufficient.json", insufficient), "train", 3
            )
        except ValueError:
            pass
        else:
            raise AssertionError("insufficient references were not rejected")
        print("insufficient reference rejection test: PASS")
        duplicate = copy.deepcopy(manifest)
        duplicate["splits"]["val"].append("cloth_red")
        try:
            ImageConditionedEpisodeDataset(
                _write_manifest(root, "duplicate.json", duplicate), "train", 2
            )
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate split clothing was not rejected")
        print("duplicate split rejection test: PASS")
        missing = copy.deepcopy(manifest)
        missing["clothes"]["cloth_red"]["frames"][0]["rgb"] = "missing.png"
        try:
            ImageConditionedEpisodeDataset(
                _write_manifest(root, "missing.json", missing), "train", 2
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing frame file was not rejected")
        print("missing file rejection test: PASS")
        if manifest_path.read_bytes() != original_manifest_bytes:
            raise AssertionError("dataset modified its input manifest")
        print("input immutability test: PASS")
        tensors = list(_all_tensors(episode))
        if any(tensor.device.type != "cpu" for tensor in tensors) or any(
            not torch.isfinite(tensor).all()
            for tensor in tensors
            if torch.is_floating_point(tensor)
        ):
            raise AssertionError("episode contains non-CPU or non-finite tensors")
        print("finite CPU output test: PASS")


if __name__ == "__main__":
    main()
