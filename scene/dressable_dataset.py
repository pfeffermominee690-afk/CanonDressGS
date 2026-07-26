from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import hashlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.dressable_camera_utils import validate_camera_data
from utils.gaussian_alignment import load_anchor_offset_target


class AnchorOffsetDataset(Dataset):
    """CPU dataset for versioned per-clothing anchor offset targets."""

    def __init__(
        self,
        target_paths: list[str | Path],
        cloth_id_map: dict[str, int] | None = None,
        anchor_tolerance: float = 1e-6,
    ) -> None:
        if not target_paths:
            raise ValueError("target_paths must contain at least one anchor target")
        if anchor_tolerance < 0:
            raise ValueError("anchor_tolerance must be non-negative")

        loaded_samples: list[dict[str, Any]] = []
        reference_anchor: torch.Tensor | None = None
        for target_path in sorted((Path(path) for path in target_paths), key=lambda path: str(path)):
            loaded = load_anchor_offset_target(target_path, device="cpu")
            target = loaded["target"]
            metadata = dict(loaded["metadata"])
            cloth_name = self._resolve_cloth_name(target_path, metadata)
            anchor_xyz = target["anchor_xyz"].detach().cpu()
            if reference_anchor is None:
                reference_anchor = anchor_xyz.clone()
            elif anchor_xyz.shape != reference_anchor.shape:
                raise ValueError(
                    f"anchor count/shape mismatch in {target_path}: "
                    f"got {tuple(anchor_xyz.shape)}, expected {tuple(reference_anchor.shape)}"
                )
            elif not torch.allclose(
                anchor_xyz,
                reference_anchor,
                atol=anchor_tolerance,
                rtol=0,
            ):
                raise ValueError(f"anchor_xyz mismatch exceeds tolerance in {target_path}")
            sample = {
                "cloth_name": cloth_name,
                "anchor_xyz": anchor_xyz,
                "delta_xyz": target["delta_xyz"].detach().cpu(),
                "delta_scaling": target["delta_scaling"].detach().cpu(),
                "delta_opacity": target["delta_opacity"].detach().cpu(),
                "valid_mask": target["valid_mask"].detach().cpu(),
                "cloth_region_weight": target["cloth_region_weight"].detach().cpu(),
                "metadata": metadata,
            }
            if "fallback_mask" in target:
                sample["fallback_mask"] = target["fallback_mask"].detach().cpu()
            loaded_samples.append(sample)

        cloth_names = sorted({sample["cloth_name"] for sample in loaded_samples})
        if cloth_id_map is None:
            resolved_map = {name: index for index, name in enumerate(cloth_names)}
        else:
            resolved_map = self._validate_cloth_id_map(cloth_id_map, cloth_names)
        for sample in loaded_samples:
            sample["cloth_id"] = torch.tensor(
                resolved_map[sample["cloth_name"]],
                dtype=torch.long,
            )

        self._samples = loaded_samples
        self._cloth_id_map = resolved_map
        self._anchor_xyz = reference_anchor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]

    @property
    def num_clothes(self) -> int:
        return len(self._cloth_id_map)

    @property
    def anchor_xyz(self) -> torch.Tensor:
        return self._anchor_xyz.clone()

    def get_cloth_id_map(self) -> dict[str, int]:
        return dict(self._cloth_id_map)

    @staticmethod
    def _resolve_cloth_name(path: Path, metadata: dict[str, Any]) -> str:
        if "cloth_name" in metadata:
            name = str(metadata["cloth_name"])
        elif "cloth_id" in metadata:
            name = str(metadata["cloth_id"])
        else:
            name = path.parent.name
        if not name:
            raise ValueError(f"could not resolve a non-empty cloth name for {path}")
        return name

    @staticmethod
    def _validate_cloth_id_map(
        cloth_id_map: dict[str, int],
        cloth_names: list[str],
    ) -> dict[str, int]:
        if not isinstance(cloth_id_map, dict):
            raise TypeError("cloth_id_map must be a dict or None")
        normalized = {str(name): cloth_id for name, cloth_id in cloth_id_map.items()}
        missing = sorted(set(cloth_names) - set(normalized))
        extra = sorted(set(normalized) - set(cloth_names))
        if missing or extra:
            raise ValueError(f"cloth_id_map names mismatch: missing={missing}, extra={extra}")
        ids = list(normalized.values())
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
            raise TypeError("all cloth ids must be Python integers")
        if sorted(ids) != list(range(len(cloth_names))):
            raise ValueError("cloth ids must be unique and contiguous from 0 to num_clothes-1")
        return normalized


class RenderingDressableDataset(Dataset):
    """CPU dataset joining RGB/masks, pose, camera, and anchor teacher targets."""

    def __init__(
        self,
        samples_json: str | Path,
        anchor_target_root: str | Path | None = None,
        cloth_id_map: dict[str, int] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.samples_json = Path(samples_json)
        if not self.samples_json.is_file():
            raise FileNotFoundError(f"rendering samples JSON does not exist: {self.samples_json}")
        with self.samples_json.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise ValueError("samples JSON must contain a non-empty list or {'samples': [...]} object")
        if image_size is not None:
            if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
                raise ValueError("image_size must be a positive (height, width) tuple")
            image_size = (int(image_size[0]), int(image_size[1]))
        self.image_size = image_size
        self.anchor_target_root = (
            None if anchor_target_root is None else Path(anchor_target_root).resolve()
        )
        self._records = [self._resolve_record(record) for record in records]
        cloth_names = sorted({record["cloth_name"] for record in self._records})
        if cloth_id_map is None:
            self._cloth_id_map = {name: index for index, name in enumerate(cloth_names)}
        else:
            self._cloth_id_map = AnchorOffsetDataset._validate_cloth_id_map(
                cloth_id_map,
                cloth_names,
            )
        self._anchor_xyz: torch.Tensor | None = None
        self.validate_samples()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        target_rgb = self._load_image(record["target_rgb"], rgb=True)
        foreground_mask = self._load_image(record["foreground_mask"], rgb=False)
        cloth_mask_path = record.get("cloth_mask")
        cloth_mask = (
            foreground_mask.clone()
            if cloth_mask_path is None
            else self._load_image(cloth_mask_path, rgb=False)
        )
        pose = torch.from_numpy(np.load(record["pose"], allow_pickle=False)).float().reshape(-1)
        if not torch.isfinite(pose).all():
            raise ValueError(f"pose contains NaN or Inf: {record['pose']}")
        with record["camera"].open("r", encoding="utf-8") as file:
            camera = json.load(file)
        validate_camera_data(camera)
        loaded_target = load_anchor_offset_target(record["anchor_offset_target"], device="cpu")
        sample = {
            "identity_id": record["identity_id"],
            "cloth_name": record["cloth_name"],
            "cloth_id": torch.tensor(
                self._cloth_id_map[record["cloth_name"]],
                dtype=torch.long,
            ),
            "frame_id": record["frame_id"],
            "view_id": record["view_id"],
            "target_rgb": target_rgb,
            "foreground_mask": foreground_mask,
            "cloth_mask": cloth_mask,
            "pose": pose,
            "camera": camera,
            "anchor_offset_target": loaded_target["target"],
            "anchor_metadata": loaded_target["metadata"],
            "Th": torch.as_tensor(record.get("Th", [0.0, 0.0, 0.0]), dtype=torch.float32),
            "Rh": torch.as_tensor(record.get("Rh", torch.eye(3).tolist()), dtype=torch.float32),
        }
        if sample["Th"].shape != (3,) or sample["Rh"].shape != (3, 3):
            raise ValueError("optional Th/Rh must have shapes [3] and [3,3]")
        return sample

    @property
    def num_clothes(self) -> int:
        return len(self._cloth_id_map)

    @property
    def anchor_xyz(self) -> torch.Tensor:
        return self._anchor_xyz.clone()

    def get_cloth_id_map(self) -> dict[str, int]:
        return dict(self._cloth_id_map)

    def validate_samples(self) -> None:
        """Eagerly validate paths, image geometry, cameras, poses, and anchors."""

        reference_anchor: torch.Tensor | None = None
        reference_size: tuple[int, int] | None = self.image_size
        for index in range(len(self._records)):
            sample = self[index]
            height, width = sample["target_rgb"].shape[-2:]
            if sample["foreground_mask"].shape != (1, height, width):
                raise ValueError("foreground mask size does not match target RGB")
            if sample["cloth_mask"].shape != (1, height, width):
                raise ValueError("cloth mask size does not match target RGB")
            if reference_size is None:
                reference_size = (height, width)
            elif (height, width) != reference_size:
                raise ValueError("all rendering samples must share one image size")
            anchor_xyz = sample["anchor_offset_target"]["anchor_xyz"]
            if reference_anchor is None:
                reference_anchor = anchor_xyz.clone()
            elif anchor_xyz.shape != reference_anchor.shape or not torch.allclose(
                anchor_xyz,
                reference_anchor,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("rendering samples use inconsistent anchor topology")
        self.image_size = reference_size
        self._anchor_xyz = reference_anchor

    def _resolve_record(self, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("every rendering sample must be a JSON object")
        required = (
            "identity_id",
            "cloth_name",
            "frame_id",
            "view_id",
            "target_rgb",
            "foreground_mask",
            "pose",
            "camera",
            "anchor_offset_target",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise KeyError(f"rendering sample is missing required keys: {missing}")
        resolved = dict(record)
        for key in ("target_rgb", "foreground_mask", "cloth_mask", "pose", "camera"):
            if key in resolved and resolved[key] is not None:
                resolved[key] = self._resolve_path(resolved[key], None)
        resolved["anchor_offset_target"] = self._resolve_path(
            resolved["anchor_offset_target"],
            self.anchor_target_root,
        )
        for key in required:
            if key in {"target_rgb", "foreground_mask", "pose", "camera", "anchor_offset_target"}:
                if not resolved[key].is_file():
                    raise FileNotFoundError(f"rendering sample path does not exist: {resolved[key]}")
        if resolved.get("cloth_mask") is not None and not resolved["cloth_mask"].is_file():
            raise FileNotFoundError(f"cloth mask does not exist: {resolved['cloth_mask']}")
        return resolved

    def _resolve_path(self, value: str | Path, root: Path | None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        base = root if root is not None else self.samples_json.parent
        return (base / path).resolve()

    def _load_image(self, path: Path, rgb: bool) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB" if rgb else "L")
            if self.image_size is not None:
                height, width = self.image_size
                resampling = Image.Resampling.BILINEAR if rgb else Image.Resampling.NEAREST
                image = image.resize((width, height), resample=resampling)
            array = np.array(image, copy=True)
        tensor = torch.from_numpy(array).float() / 255.0
        if rgb:
            return tensor.permute(2, 0, 1).contiguous()
        return (tensor >= 0.5).float().unsqueeze(0).contiguous()


class ImageConditionedEpisodeDataset(Dataset):
    """Sample same-clothing reference sets with a disjoint target observation."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        reference_count: int | tuple[int, int],
        image_size: tuple[int, int] | None = None,
        random_reference_sampling: bool = True,
        exclude_target_from_reference: bool = True,
        seed: int = 0,
        anchor_target_root: str | Path | None = None,
        allow_clothing_mask_fallback: bool = True,
        allow_missing_rh_th: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"episode manifest does not exist: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise TypeError("episode manifest root must be an object")
        self.manifest = manifest
        self.split = str(split)
        self.reference_count_range = self._parse_reference_count(reference_count)
        self.random_reference_sampling = bool(random_reference_sampling)
        self.exclude_target_from_reference = bool(exclude_target_from_reference)
        if not self.exclude_target_from_reference:
            raise ValueError(
                "the first image-conditioned dataset requires target/reference separation"
            )
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.allow_clothing_mask_fallback = bool(allow_clothing_mask_fallback)
        self.allow_missing_rh_th = bool(allow_missing_rh_th)
        self.anchor_target_root = (
            None if anchor_target_root is None else Path(anchor_target_root).resolve()
        )
        if image_size is not None:
            if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
                raise ValueError("image_size must be a positive (height,width) tuple")
            image_size = (int(image_size[0]), int(image_size[1]))
        self.image_size = image_size
        self._cloth_id_map = {
            name: index for index, name in enumerate(sorted(manifest.get("clothes", {})))
        }
        self._resolved_clothes: dict[str, dict[str, Any]] = {}
        self._episodes: list[tuple[str, int]] = []
        self._anchor_xyz: torch.Tensor | None = None
        self.pose_dim = 0
        self.identity_id = str(manifest.get("identity_id", ""))
        self.validate_manifest()
        split_clothes = manifest["splits"][self.split]
        for cloth_name in sorted(split_clothes):
            frames = self._resolved_clothes[cloth_name]["frames"]
            self._episodes.extend((cloth_name, index) for index in range(len(frames)))
        self.manifest_fingerprint = self._compute_manifest_fingerprint()

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        deterministic = self.split != "train" or not self.random_reference_sampling
        return self.sample_episode(index, deterministic=deterministic)

    def sample_episode(
        self,
        index: int,
        sampling_salt: int = 0,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Build one episode; a salt provides an independent reference subset."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("episode index must be an int")
        if index < 0:
            index += len(self._episodes)
        if not 0 <= index < len(self._episodes):
            raise IndexError("episode index out of range")
        cloth_name, target_index = self._episodes[index]
        cloth = self._resolved_clothes[cloth_name]
        frames = cloth["frames"]
        candidates = [
            frame_index
            for frame_index in range(len(frames))
            if not self.exclude_target_from_reference or frame_index != target_index
        ]
        rng = (
            random.Random(self.seed + index * 1000003 + int(sampling_salt))
            if deterministic or self.split != "train"
            else self._rng
        )
        minimum, maximum = self.reference_count_range
        count = minimum if minimum == maximum else rng.randint(minimum, maximum)
        if len(candidates) < count:
            raise ValueError(
                f"cloth {cloth_name!r} has {len(candidates)} eligible references, "
                f"but episode requires K={count}"
            )
        reference_indices = rng.sample(candidates, count)
        rng.shuffle(reference_indices)
        references = [self._load_frame(frames[item]) for item in reference_indices]
        target = self._load_frame(frames[target_index])
        anchor_target = None
        anchor_metadata: dict[str, Any] = {}
        if cloth["anchor_offset_target"] is not None:
            loaded = load_anchor_offset_target(cloth["anchor_offset_target"], device="cpu")
            anchor_target = loaded["target"]
            anchor_metadata = loaded["metadata"]
        episode = {
            "identity_id": self.identity_id,
            "cloth_name": cloth_name,
            "cloth_id": torch.tensor(self._cloth_id_map[cloth_name], dtype=torch.long),
            "episode_index": index,
            "reference_images": torch.stack([item["rgb"] for item in references]),
            "reference_cloth_masks": torch.stack(
                [item["clothing_mask"] for item in references]
            ),
            "reference_foreground_masks": torch.stack(
                [item["foreground_mask"] for item in references]
            ),
            "reference_poses": torch.stack([item["pose"] for item in references]),
            "reference_Rh": torch.stack([item["Rh"] for item in references]),
            "reference_Th": torch.stack([item["Th"] for item in references]),
            "reference_cameras": [item["camera"] for item in references],
            "reference_frame_ids": [item["frame_id"] for item in references],
            "reference_view_ids": [item["view_id"] for item in references],
            "reference_valid_mask": torch.ones(count, dtype=torch.float32),
            "target_rgb": target["rgb"],
            "target_foreground_mask": target["foreground_mask"],
            "target_clothing_mask": target["clothing_mask"],
            "target_pose": target["pose"],
            "target_Rh": target["Rh"],
            "target_Th": target["Th"],
            "target_camera": target["camera"],
            "target_frame_id": target["frame_id"],
            "target_view_id": target["view_id"],
            "anchor_offset_target": anchor_target,
            "anchor_metadata": anchor_metadata,
            "metadata": {
                "reference_clothing_mask_fallback": [
                    item["clothing_mask_fallback"] for item in references
                ],
                "target_clothing_mask_fallback": target["clothing_mask_fallback"],
                "reference_rh_th_fallback": [
                    item["rh_th_fallback"] for item in references
                ],
                "target_rh_th_fallback": target["rh_th_fallback"],
            },
        }
        self._validate_episode_output(episode)
        return episode

    @property
    def num_clothes(self) -> int:
        return len(self._cloth_id_map)

    @property
    def anchor_xyz(self) -> torch.Tensor:
        if self._anchor_xyz is None:
            raise RuntimeError("manifest does not provide anchor targets")
        return self._anchor_xyz.clone()

    @property
    def has_anchor_targets(self) -> bool:
        return self._anchor_xyz is not None

    def get_anchor_xyz_or_none(self) -> torch.Tensor | None:
        return None if self._anchor_xyz is None else self._anchor_xyz.clone()

    def get_cloth_id_map(self) -> dict[str, int]:
        return dict(self._cloth_id_map)

    def validate_manifest(self) -> None:
        """Eagerly check split isolation, paths, frame tensors, and anchor topology."""

        splits = self.manifest.get("splits")
        clothes = self.manifest.get("clothes")
        if not self.identity_id:
            raise ValueError("manifest identity_id must be non-empty")
        if not isinstance(splits, dict) or not isinstance(clothes, dict) or not clothes:
            raise ValueError("manifest requires non-empty splits and clothes objects")
        if self.split not in splits:
            raise KeyError(f"manifest does not define split {self.split!r}")
        memberships: dict[str, str] = {}
        for split_name, names in splits.items():
            if not isinstance(names, list):
                raise TypeError(f"split {split_name!r} must be a list")
            for name in names:
                if name not in clothes:
                    raise KeyError(f"split {split_name!r} references unknown cloth {name!r}")
                if name in memberships:
                    raise ValueError(
                        f"cloth {name!r} occurs in both {memberships[name]!r} and {split_name!r}"
                    )
                memberships[name] = split_name
        if set(memberships) != set(clothes):
            missing = sorted(set(clothes) - set(memberships))
            raise ValueError(f"clothes missing from split assignment: {missing}")

        reference_anchor = None
        reference_size = self.image_size
        pose_dim = None
        for cloth_name, raw_cloth in clothes.items():
            if not isinstance(raw_cloth, dict):
                raise TypeError(f"cloth {cloth_name!r} must be an object")
            raw_frames = raw_cloth.get("frames")
            if not isinstance(raw_frames, list) or not raw_frames:
                raise ValueError(f"cloth {cloth_name!r} requires non-empty frames")
            required_references = self.reference_count_range[1]
            eligible_count = len(raw_frames) - 1
            if eligible_count < required_references:
                raise ValueError(
                    f"cloth {cloth_name!r} has insufficient frames for K={required_references}"
                )
            resolved_frames = [self._resolve_frame(frame, cloth_name) for frame in raw_frames]
            anchor_path = raw_cloth.get("anchor_offset_target")
            resolved_anchor = (
                None
                if anchor_path is None
                else self._resolve_path(anchor_path, self.anchor_target_root)
            )
            if resolved_anchor is not None and not resolved_anchor.is_file():
                raise FileNotFoundError(f"anchor target does not exist: {resolved_anchor}")
            self._resolved_clothes[cloth_name] = {
                "frames": resolved_frames,
                "anchor_offset_target": resolved_anchor,
            }
            for frame in resolved_frames:
                loaded = self._load_frame(frame)
                current_size = tuple(loaded["rgb"].shape[-2:])
                if reference_size is None:
                    reference_size = current_size
                elif current_size != reference_size:
                    raise ValueError("all manifest images must share one output image size")
                current_pose_dim = loaded["pose"].numel()
                if pose_dim is None:
                    pose_dim = current_pose_dim
                elif current_pose_dim != pose_dim:
                    raise ValueError("all manifest poses must have the same shape")
            if resolved_anchor is not None:
                anchor = load_anchor_offset_target(resolved_anchor, device="cpu")["target"][
                    "anchor_xyz"
                ]
                if reference_anchor is None:
                    reference_anchor = anchor.clone()
                elif anchor.shape != reference_anchor.shape or not torch.allclose(
                    anchor, reference_anchor, atol=1e-6, rtol=0
                ):
                    raise ValueError("manifest anchor targets use inconsistent topology")
        self.image_size = reference_size
        self.pose_dim = int(pose_dim)
        self._anchor_xyz = reference_anchor

    def _resolve_frame(self, frame: Any, cloth_name: str) -> dict[str, Any]:
        if not isinstance(frame, dict):
            raise TypeError(f"frames for {cloth_name!r} must be objects")
        required = (
            "frame_id",
            "view_id",
            "rgb",
            "foreground_mask",
            "pose",
            "camera",
        )
        missing = [key for key in required if key not in frame]
        if missing:
            raise KeyError(f"frame for {cloth_name!r} is missing {missing}")
        resolved = dict(frame)
        for key in ("rgb", "foreground_mask", "clothing_mask", "pose", "camera"):
            if resolved.get(key) is not None:
                resolved[key] = self._resolve_path(resolved[key], None)
        for key in ("rgb", "foreground_mask", "pose", "camera"):
            if not resolved[key].is_file():
                raise FileNotFoundError(f"manifest frame path does not exist: {resolved[key]}")
        if resolved.get("clothing_mask") is not None and not resolved[
            "clothing_mask"
        ].is_file():
            raise FileNotFoundError(
                f"manifest clothing mask does not exist: {resolved['clothing_mask']}"
            )
        if resolved.get("clothing_mask") is None and not self.allow_clothing_mask_fallback:
            raise ValueError("clothing_mask is missing and fallback is disabled")
        has_rh = resolved.get("Rh") is not None
        has_th = resolved.get("Th") is not None
        if has_rh != has_th:
            raise ValueError("Rh and Th must either both be provided or both be absent")
        if not has_rh:
            if not self.allow_missing_rh_th:
                raise ValueError(
                    "real image-conditioned frames require explicit Rh and Th"
                )
            resolved["Rh"] = None
            resolved["Th"] = None
        else:
            for key in ("Rh", "Th"):
                value = resolved[key]
                if isinstance(value, (str, Path)):
                    path = self._resolve_path(value, None)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"manifest frame transform does not exist: {path}"
                        )
                    if path.suffix.lower() != ".npy":
                        raise ValueError(f"{key} path must point to a .npy file")
                    resolved[key] = path
        return resolved

    def _load_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        rgb = self._load_image(frame["rgb"], rgb=True)
        foreground = self._load_image(frame["foreground_mask"], rgb=False)
        fallback = frame.get("clothing_mask") is None
        clothing = (
            foreground.clone()
            if fallback
            else self._load_image(frame["clothing_mask"], rgb=False)
        )
        pose_array = np.load(frame["pose"], allow_pickle=False)
        pose = torch.from_numpy(np.asarray(pose_array)).float().reshape(-1)
        if pose.numel() == 0 or not torch.isfinite(pose).all():
            raise ValueError(f"pose must be non-empty and finite: {frame['pose']}")
        with frame["camera"].open("r", encoding="utf-8") as file:
            camera = json.load(file)
        validate_camera_data(camera)
        Rh, Th, rh_th_fallback = self._load_rh_th(frame)
        return {
            "frame_id": str(frame["frame_id"]),
            "view_id": str(frame["view_id"]),
            "rgb": rgb,
            "foreground_mask": foreground,
            "clothing_mask": clothing,
            "clothing_mask_fallback": fallback,
            "pose": pose,
            "Rh": Rh,
            "Th": Th,
            "rh_th_fallback": rh_th_fallback,
            "camera": camera,
        }

    def _load_rh_th(
        self,
        frame: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        fallback = frame["Rh"] is None
        if fallback:
            return (
                torch.eye(3, dtype=torch.float32),
                torch.zeros(3, dtype=torch.float32),
                True,
            )
        return (
            self._load_transform(frame["Rh"], "Rh", (3, 3)),
            self._load_transform(frame["Th"], "Th", (3,)),
            False,
        )

    def _rh_th_fingerprint_payload(self, frame: dict[str, Any]) -> dict[str, Any]:
        Rh, Th, _ = self._load_rh_th(frame)
        return {"Rh": Rh.tolist(), "Th": Th.tolist()}

    @staticmethod
    def _load_transform(
        value: Path | list[Any],
        name: str,
        expected_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if isinstance(value, Path):
            array = np.load(value, allow_pickle=False)
            tensor = torch.from_numpy(np.asarray(array)).float()
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return tensor.contiguous()

    def _load_image(self, path: Path, rgb: bool) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB" if rgb else "L")
            if self.image_size is not None:
                height, width = self.image_size
                resampling = Image.Resampling.BILINEAR if rgb else Image.Resampling.NEAREST
                image = image.resize((width, height), resample=resampling)
            array = np.array(image, copy=True)
        tensor = torch.from_numpy(array).float() / 255
        if rgb:
            return tensor.permute(2, 0, 1).contiguous()
        return tensor.unsqueeze(0).contiguous().clamp(0, 1)

    def _resolve_path(self, value: str | Path, root: Path | None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        base = root if root is not None else self.manifest_path.parent
        return (base / path).resolve()

    def _compute_manifest_fingerprint(self) -> str:
        frame_counts = {
            name: len(self._resolved_clothes[name]["frames"])
            for name in self.manifest["splits"][self.split]
        }
        payload = {
            "manifest": self.manifest,
            "split": self.split,
            "split_clothes": sorted(self.manifest["splits"][self.split]),
            "frame_counts": frame_counts,
            "image_size": self.image_size,
            "pose_dim": self.pose_dim,
            "rh_th": {
                name: [
                    self._rh_th_fingerprint_payload(frame)
                    for frame in self._resolved_clothes[name]["frames"]
                ]
                for name in self.manifest["splits"][self.split]
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_reference_count(value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, int) and not isinstance(value, bool):
            minimum = maximum = value
        elif isinstance(value, tuple) and len(value) == 2:
            minimum, maximum = value
        else:
            raise TypeError("reference_count must be an int or (K_min,K_max) tuple")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in (minimum, maximum)):
            raise TypeError("reference counts must be integers")
        if minimum <= 0 or maximum < minimum:
            raise ValueError("reference counts must satisfy 0 < K_min <= K_max")
        return minimum, maximum

    @staticmethod
    def _validate_episode_output(episode: dict[str, Any]) -> None:
        reference_ids = set(zip(episode["reference_frame_ids"], episode["reference_view_ids"]))
        target_id = (episode["target_frame_id"], episode["target_view_id"])
        if target_id in reference_ids:
            raise RuntimeError("target observation leaked into reference set")
        tensors = [value for value in episode.values() if isinstance(value, torch.Tensor)]
        if any(value.device.type != "cpu" for value in tensors):
            raise RuntimeError("episode dataset must keep all tensors on CPU")
        floating = [value for value in tensors if torch.is_floating_point(value)]
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("episode contains NaN or Inf")
        reference_images = episode["reference_images"]
        if reference_images.ndim != 4:
            raise ValueError("reference_images must have shape [K,3,H,W]")
        num_views = reference_images.shape[0]
        for key in (
            "reference_cloth_masks",
            "reference_foreground_masks",
            "reference_poses",
            "reference_valid_mask",
        ):
            value = episode.get(key)
            if not isinstance(value, torch.Tensor) or value.shape[0] != num_views:
                raise ValueError(f"{key} first dimension must match K={num_views}")
        for key in ("reference_frame_ids", "reference_view_ids"):
            if len(episode.get(key, ())) != num_views:
                raise ValueError(f"{key} length must match K={num_views}")
        expected = {
            "reference_Rh": (num_views, 3, 3),
            "reference_Th": (num_views, 3),
            "target_Rh": (3, 3),
            "target_Th": (3,),
        }
        for key, shape in expected.items():
            value = episode.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{key} must have shape {shape}")
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"{key} must be a finite floating-point tensor")
        if len(episode["reference_cameras"]) != num_views:
            raise ValueError("reference camera count must match K")


def image_conditioned_episode_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve nested camera/reference structures for the batch-size-one path."""

    if len(batch) != 1:
        raise ValueError("image-conditioned episode training currently requires batch_size=1")
    return batch[0]
