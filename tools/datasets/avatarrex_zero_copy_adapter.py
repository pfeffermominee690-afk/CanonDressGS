#!/usr/bin/env python3
"""Read-only AvatarReX adapter and deterministic split builder.

The adapter never copies or rewrites RGB, masks, calibration, or SMPL-X data.
Only compact manifests under an explicitly supplied output path may be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_ARCHIVE_SHA256 = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
EXPECTED_RAW_FULL_CONTENT_FINGERPRINT = "00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15"
EXPECTED_CAMERA_COUNT = 16
EXPECTED_FRAME_COUNT = 1901
SEED = 20260723
TEMPORAL_BUFFER_RADIUS = 5
MASK_RELATIVE_DIRECTORY = Path("mask") / "pha"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class AvatarReXZeroCopyAdapter:
    """Canonical pointer view over an untouched ``avatarrex_lbn1`` root."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root.resolve()
        self.calibration_path = self.raw_root / "calibration_full.json"
        self.smpl_path = self.raw_root / "smpl_params.npz"
        if not self.raw_root.is_dir():
            raise FileNotFoundError(self.raw_root)
        if not self.calibration_path.is_file() or not self.smpl_path.is_file():
            raise FileNotFoundError("AvatarReX root lacks calibration_full.json or smpl_params.npz")

    def camera_names(self) -> list[str]:
        names = sorted(path.name for path in self.raw_root.iterdir() if path.is_dir() and path.name.isdigit())
        if len(names) != EXPECTED_CAMERA_COUNT:
            raise ValueError(f"expected {EXPECTED_CAMERA_COUNT} cameras, found {len(names)}")
        return names

    def camera_mapping(self) -> list[dict[str, Any]]:
        return [
            {"raw_camera_name": name, "canonical_camera_id": f"camera_{index:02d}", "camera_index": index}
            for index, name in enumerate(self.camera_names())
        ]

    @staticmethod
    def _frame_name(frame_id: int) -> str:
        if not 0 <= frame_id < EXPECTED_FRAME_COUNT:
            raise IndexError(frame_id)
        return f"{frame_id:08d}"

    def resolve_rgb(self, raw_camera_name: str, frame_id: int) -> Path:
        if raw_camera_name not in self.camera_names():
            raise KeyError(raw_camera_name)
        return self.raw_root / raw_camera_name / f"{self._frame_name(frame_id)}.jpg"

    def mask_suffix(self, raw_camera_name: str) -> str:
        mask_dir = self.raw_root / raw_camera_name / MASK_RELATIVE_DIRECTORY
        if not mask_dir.is_dir():
            raise FileNotFoundError(mask_dir)
        suffixes = sorted({path.suffix.lower() for path in mask_dir.iterdir() if path.is_file()})
        if len(suffixes) != 1:
            raise ValueError(f"ambiguous mask suffixes for {raw_camera_name}: {suffixes}")
        return suffixes[0]

    def resolve_mask(self, raw_camera_name: str, frame_id: int) -> Path:
        if raw_camera_name not in self.camera_names():
            raise KeyError(raw_camera_name)
        return (
            self.raw_root
            / raw_camera_name
            / MASK_RELATIVE_DIRECTORY
            / f"{self._frame_name(frame_id)}{self.mask_suffix(raw_camera_name)}"
        )

    def load_calibration(self) -> dict[str, Any]:
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        if sorted(payload) != self.camera_names():
            raise ValueError("calibration cameras do not match camera directories")
        return payload

    def availability_manifest(self) -> dict[str, Any]:
        camera_rows = []
        all_expected = {f"{frame:08d}" for frame in range(EXPECTED_FRAME_COUNT)}
        for mapping in self.camera_mapping():
            name = mapping["raw_camera_name"]
            camera_root = self.raw_root / name
            rgb_ids = {path.stem for path in camera_root.glob("*.jpg")}
            mask_dir = camera_root / MASK_RELATIVE_DIRECTORY
            mask_ids = {path.stem for path in mask_dir.iterdir() if path.is_file()}
            camera_rows.append(
                {
                    **mapping,
                    "rgb_count": len(rgb_ids),
                    "mask_count": len(mask_ids),
                    "rgb_path_template": str(camera_root / "{frame_id:08d}.jpg"),
                    "mask_path_template": str(mask_dir / ("{frame_id:08d}" + self.mask_suffix(name))),
                    "missing_rgb_frame_ids": sorted(all_expected - rgb_ids),
                    "missing_mask_frame_ids": sorted(all_expected - mask_ids),
                    "extra_rgb_frame_ids": sorted(rgb_ids - all_expected),
                    "extra_mask_frame_ids": sorted(mask_ids - all_expected),
                }
            )
        return {
            "representation": "COMPRESSED_ZERO_COPY_FRAME_MANIFEST",
            "frame_id_range_inclusive": [0, EXPECTED_FRAME_COUNT - 1],
            "expected_frame_count_per_camera": EXPECTED_FRAME_COUNT,
            "camera_rows": camera_rows,
            "logical_record_count": EXPECTED_CAMERA_COUNT * EXPECTED_FRAME_COUNT,
        }

    def camera_split(self) -> dict[str, Any]:
        import numpy as np

        calibration = self.load_calibration()
        entries: list[dict[str, Any]] = []
        raw_centers = []
        mappings = self.camera_mapping()
        for mapping in mappings:
            name = mapping["raw_camera_name"]
            camera = calibration[name]
            rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
            translation = np.asarray(camera["T"], dtype=np.float64).reshape(3)
            center = -(rotation.T @ translation)
            raw_centers.append(center)
            entries.append({**mapping, "center": center.tolist()})
        rig_center = np.mean(np.stack(raw_centers), axis=0)
        for entry in entries:
            relative = np.asarray(entry["center"]) - rig_center
            entry["relative_center"] = relative.tolist()
            entry["azimuth_degrees"] = float(math.degrees(math.atan2(relative[0], relative[2])))
            entry["elevation_degrees"] = float(
                math.degrees(math.atan2(relative[1], math.hypot(relative[0], relative[2])))
            )
        ordered = sorted(entries, key=lambda row: (row["azimuth_degrees"], row["camera_index"]))
        heldout_ranks = list(range(0, EXPECTED_CAMERA_COUNT, 4))
        heldout_ids = sorted(ordered[rank]["canonical_camera_id"] for rank in heldout_ranks)
        all_ids = {row["canonical_camera_id"] for row in entries}
        train_ids = sorted(all_ids - set(heldout_ids))
        split_payload = {
            "algorithm": "camera center C=-R^T T; subtract mean rig center; sort by (azimuth,camera_index); select ranks 0,4,8,12",
            "tie_rule": "ascending camera_index",
            "train_camera_ids": train_ids,
            "heldout_camera_ids": heldout_ids,
        }
        return {
            "camera_count": EXPECTED_CAMERA_COUNT,
            "rig_center": rig_center.tolist(),
            "cameras": entries,
            "azimuth_order_camera_ids": [row["canonical_camera_id"] for row in ordered],
            "heldout_ranks": heldout_ranks,
            **split_payload,
            "train_count": len(train_ids),
            "heldout_count": len(heldout_ids),
            "overlap": sorted(set(train_ids) & set(heldout_ids)),
            "split_sha256": canonical_sha256(split_payload),
        }

    def pose_split(self) -> dict[str, Any]:
        import numpy as np
        from scipy.spatial.transform import Rotation

        with np.load(self.smpl_path, allow_pickle=False) as payload:
            body = np.asarray(payload["body_pose"], dtype=np.float64)
            global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
        frame_count = int(body.shape[0])
        heldout_target = round(frame_count * 0.05)

        def rotation_6d(axis_angle: Any) -> Any:
            flat = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
            matrices = Rotation.from_rotvec(flat).as_matrix()
            return matrices[:, :, :2].reshape(*axis_angle.shape[:-1], 6)

        def standardize(values: Any) -> tuple[Any, int]:
            std = values.std(axis=0)
            active = std > 1e-8
            normalized = (values[:, active] - values[:, active].mean(axis=0)) / std[active]
            return normalized, int(active.sum())

        body6 = rotation_6d(body.reshape(frame_count, -1, 3)).reshape(frame_count, -1)
        global6 = rotation_6d(global_orient.reshape(frame_count, 1, 3)).reshape(frame_count, -1)
        body_z, body_dims = standardize(body6)
        global_z, global_dims = standardize(global6)
        descriptor = np.concatenate([body_z, 0.25 * global_z], axis=1).astype(np.float32)
        rng = np.random.default_rng(SEED)
        first = int(rng.integers(0, frame_count))
        selected = [first]
        minimum_distance_sq = np.sum((descriptor - descriptor[first]) ** 2, axis=1)
        while len(selected) < heldout_target:
            allowed = np.ones(frame_count, dtype=bool)
            for chosen in selected:
                allowed[max(0, chosen - 10) : min(frame_count, chosen + 11)] = False
            scores = np.where(allowed, minimum_distance_sq, -np.inf)
            chosen = int(np.argmax(scores))
            if not np.isfinite(scores[chosen]):
                raise RuntimeError("unable to select temporally separated AvatarReX poses")
            selected.append(chosen)
            distance_sq = np.sum((descriptor - descriptor[chosen]) ** 2, axis=1)
            minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
        heldout = sorted(selected)
        heldout_set = set(heldout)
        buffered = set()
        for frame in heldout:
            buffered.update(
                range(
                    max(0, frame - TEMPORAL_BUFFER_RADIUS),
                    min(frame_count, frame + TEMPORAL_BUFFER_RADIUS + 1),
                )
            )
        buffer_only = sorted(buffered - heldout_set)
        train = sorted(set(range(frame_count)) - buffered)
        split_payload = {
            "algorithm": "seeded farthest-point sampling in standardized rotation-6D pose space with >=11-frame held-out spacing",
            "seed": SEED,
            "global_orientation_handling": "standardized separately and concatenated at weight 0.25",
            "temporal_buffer_radius": TEMPORAL_BUFFER_RADIUS,
            "train_frame_ids": train,
            "heldout_frame_ids": heldout,
            "buffer_excluded_frame_ids": buffer_only,
        }
        return {
            "total_frames": frame_count,
            "valid_frames": frame_count,
            "heldout_target_fraction": 0.05,
            "body_descriptor_active_dimensions": body_dims,
            "global_descriptor_active_dimensions": global_dims,
            **split_payload,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "buffer_excluded_count": len(buffer_only),
            "train_heldout_overlap": sorted(set(train) & heldout_set),
            "train_buffer_overlap": sorted(set(train) & set(buffer_only)),
            "heldout_pair_min_temporal_distance": min(
                abs(left - right)
                for index, left in enumerate(heldout)
                for right in heldout[index + 1 :]
            ),
            "temporal_leakage": 0,
            "split_sha256": canonical_sha256(split_payload),
        }

    def smpl_schema(self) -> dict[str, Any]:
        import numpy as np

        with np.load(self.smpl_path, allow_pickle=False) as payload:
            fields = {
                key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
                for key in payload.files
            }
        return {
            "source": str(self.smpl_path),
            "source_sha256": sha256_file(self.smpl_path),
            "load_mode": "READ_ONLY_IN_MEMORY_NPZ",
            "fields": fields,
            "canonical_mapping": {
                "identity_shape": "betas[0]",
                "root_orientation": "global_orient[frame_id]",
                "translation": "transl[frame_id]",
                "body_pose": "body_pose[frame_id]",
                "jaw_pose": "jaw_pose[frame_id]",
                "expression": "expression[frame_id]",
                "left_hand_pose": "left_hand_pose[frame_id]",
                "right_hand_pose": "right_hand_pose[frame_id]",
            },
        }

    def calibration_schema(self) -> dict[str, Any]:
        calibration = self.load_calibration()
        field_sets = {name: sorted(value) for name, value in calibration.items()}
        return {
            "source": str(self.calibration_path),
            "source_sha256": sha256_file(self.calibration_path),
            "conversion_mode": "IN_MEMORY_ONLY",
            "source_fields_by_camera": field_sets,
            "canonical_mapping": {
                "intrinsics": "K",
                "world_to_camera_rotation": "R",
                "world_to_camera_translation": "T",
                "distortion_coefficients": "distCoeff",
                "image_size": "imgSize",
                "rectification_alpha": "rectifyAlpha",
                "camera_center": "-transpose(R) @ T",
            },
        }

    def loader_smoke(self) -> dict[str, Any]:
        from PIL import Image

        names = self.camera_names()
        selected_cameras = [names[0], names[len(names) // 2], names[-1]]
        selected_frames = [0, EXPECTED_FRAME_COUNT // 2, EXPECTED_FRAME_COUNT - 1]
        rows = []
        for camera in selected_cameras:
            for frame in selected_frames:
                rgb_path = self.resolve_rgb(camera, frame)
                mask_path = self.resolve_mask(camera, frame)
                with Image.open(rgb_path) as rgb:
                    rgb.load()
                    rgb_info = {"size": list(rgb.size), "mode": rgb.mode}
                with Image.open(mask_path) as mask:
                    mask.load()
                    mask_info = {"size": list(mask.size), "mode": mask.mode}
                rows.append(
                    {
                        "camera": camera,
                        "frame_id": frame,
                        "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path),
                        "rgb": rgb_info,
                        "mask": mask_info,
                        "same_dimensions": rgb_info["size"] == mask_info["size"],
                    }
                )
        return {
            "status": "PASS" if all(row["same_dimensions"] for row in rows) else "FAIL",
            "decoded_record_count": len(rows),
            "rows": rows,
            "rgb_copies": 0,
            "mask_copies": 0,
            "raw_writes": 0,
        }


def build_outputs(adapter: AvatarReXZeroCopyAdapter) -> tuple[dict[str, Any], dict[str, Any]]:
    availability = adapter.availability_manifest()
    camera = adapter.camera_split()
    pose = adapter.pose_split()
    smoke = adapter.loader_smoke()
    all_available = all(
        row["rgb_count"] == EXPECTED_FRAME_COUNT
        and row["mask_count"] == EXPECTED_FRAME_COUNT
        and not row["missing_rgb_frame_ids"]
        and not row["missing_mask_frame_ids"]
        for row in availability["camera_rows"]
    )
    split = {
        "schema_version": "avatarrex.camera_pose_split.v1",
        "task_id": "AAAI27-MULTI-IDENTITY-GARMENT-DATA-PREPARATION-001",
        "identity_id": "avatarrex_lbn1",
        "status": "PASS" if all_available and smoke["status"] == "PASS" else "FAIL",
        "source_root": str(adapter.raw_root),
        "camera_split": camera,
        "pose_split": pose,
        "availability_manifest": availability,
        "combined_protocol": {
            "train": "train cameras x train poses only",
            "strict_novel_view": "held-out cameras x train poses",
            "strict_novel_pose": "train cameras x held-out poses",
            "strict_novel_pose_and_view": "held-out cameras x held-out poses",
            "buffer_frames_forbidden_everywhere": True,
        },
        "split_content_sha256": canonical_sha256({"camera": camera, "pose": pose, "availability": availability}),
        "paper_final": 0,
    }
    contract = {
        "schema_version": "avatarrex.zero_copy_adapter_contract.v1",
        "task_id": "AAAI27-MULTI-IDENTITY-GARMENT-DATA-PREPARATION-001",
        "identity_id": "avatarrex_lbn1",
        "status": "PASS" if split["status"] == "PASS" else "FAIL",
        "standardization_mode": "READ_ONLY_ZERO_COPY_LOADER_ADAPTER",
        "source_root": str(adapter.raw_root),
        "frozen_provenance": {
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_rehash_status": "ARCHIVE_NOT_PRESENT_AT_EXTRACTED_RAW_ROOT",
            "raw_full_content_fingerprint": EXPECTED_RAW_FULL_CONTENT_FINGERPRINT,
            "raw_fingerprint_status": "FROZEN_SOURCE_CONTRACT_VALUE_NOT_RECOMPUTED_BY_LOADER_SMOKE",
        },
        "camera_mapping": adapter.camera_mapping(),
        "rgb_resolver": "{raw_root}/{raw_camera_name}/{frame_id:08d}.jpg",
        "mask_resolver": "{raw_root}/{raw_camera_name}/mask/pha/{frame_id:08d}{source_mask_suffix}",
        "calibration_adapter": adapter.calibration_schema(),
        "smpl_x_adapter": adapter.smpl_schema(),
        "canonical_schema_view": {
            "record_key": ["identity_id", "canonical_camera_id", "frame_id"],
            "rgb_and_mask": "absolute source pointers only",
            "camera": "canonical in-memory view over calibration_full.json",
            "smpl_x": "per-frame canonical view over smpl_params.npz",
        },
        "loader_smoke": smoke,
        "license_and_provenance_guard": {
            "license_file_at_raw_root": False,
            "status": "UPSTREAM_LICENSE_CONFIRMATION_REQUIRED_BEFORE_REDISTRIBUTION_OR_GENERATION",
            "raw_redistribution_authorized": False,
            "derived_generation_authorized_by_this_contract": False,
        },
        "counts": {
            "full_rgb_copy_count": 0,
            "full_mask_copy_count": 0,
            "raw_data_mutation": 0,
            "calibration_rewrites": 0,
            "smpl_npz_rewrites": 0,
        },
        "readiness": [
            "AVATARREX_ZERO_COPY_STANDARDIZATION_READY",
            "AVATARREX_BASE_AVATAR_PREPARATION_REQUIRED",
        ],
        "paper_final": 0,
    }
    return contract, split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()
    contract, split = build_outputs(AvatarReXZeroCopyAdapter(args.raw_root))
    write_json(args.contract_output, contract)
    write_json(args.split_output, split)
    print(
        json.dumps(
            {
                "status": contract["status"],
                "camera_count": split["camera_split"]["camera_count"],
                "frame_count": split["pose_split"]["total_frames"],
                "loader_smoke": contract["loader_smoke"]["status"],
                "rgb_copies": 0,
                "mask_copies": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if contract["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
