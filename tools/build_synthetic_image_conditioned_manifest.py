from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gaussian_alignment import save_anchor_offset_target


def build_synthetic_image_conditioned_manifest(output_dir: str | Path) -> Path:
    """Create a deterministic four-clothing reference-target test dataset."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    clothes = {
        "cloth_red": ("red", [0.06, 0.00, 0.00]),
        "cloth_blue": ("blue", [0.00, 0.05, 0.00]),
        "cloth_green": ("green", [0.00, 0.00, 0.04]),
        "cloth_yellow": ("yellow", [0.03, 0.03, 0.00]),
    }
    anchor_xyz = torch.stack(
        (
            torch.linspace(-0.25, 0.25, 8),
            torch.zeros(8),
            torch.full((8,), 2.0),
        ),
        dim=1,
    )
    manifest = {
        "identity_id": "synthetic_subject",
        "splits": {
            "train": ["cloth_red", "cloth_blue"],
            "val": ["cloth_green"],
            "test": ["cloth_yellow"],
        },
        "clothes": {},
    }
    for cloth_index, (cloth_name, (pattern, offset)) in enumerate(clothes.items()):
        cloth_dir = root / cloth_name
        cloth_dir.mkdir(parents=True, exist_ok=True)
        region = torch.linspace(0.2, 1.0, 8)[:, None]
        target = {
            "anchor_xyz": anchor_xyz,
            "delta_xyz": region * torch.tensor(offset)[None, :],
            "delta_scaling": region
            * torch.full((1, 3), 0.01 + 0.003 * cloth_index),
            "delta_opacity": region * (0.04 + 0.01 * cloth_index),
            "valid_mask": torch.ones(8, 1),
            "cloth_region_weight": region,
        }
        target_path = cloth_dir / "anchor_offsets.pt"
        save_anchor_offset_target(
            target,
            target_path,
            metadata={"cloth_name": cloth_name, "source": "synthetic_episode"},
        )
        frames = []
        for frame_index in range(6):
            height = width = 64
            image = np.full((height, width, 3), 18, dtype=np.uint8)
            foreground = np.zeros((height, width), dtype=np.uint8)
            foreground[5:60, 12:52] = 255
            clothing = np.zeros_like(foreground)
            shift = frame_index % 3 - 1
            if pattern == "red":
                clothing[10 + shift : 38 + shift, 17:47] = 255
                image[clothing > 0] = [220, 35 + frame_index, 30]
            elif pattern == "blue":
                clothing[12:52, 16 + shift : 48 + shift] = 255
                image[clothing > 0] = [25, 45, 215]
                image[12:52:4, 16 + shift : 48 + shift] = [70, 120, 240]
            elif pattern == "green":
                clothing[30 + shift : 56 + shift, 16:48] = 255
                image[clothing > 0] = [35, 205, 55]
            else:
                clothing[14:50, 17 + shift : 47 + shift] = 255
                image[clothing > 0] = [220, 205, 35]
            image[(foreground > 0) & (clothing == 0)] = [145, 145, 145]
            rgb_path = cloth_dir / f"rgb_{frame_index:02d}.png"
            foreground_path = cloth_dir / f"foreground_{frame_index:02d}.png"
            clothing_path = cloth_dir / f"clothing_{frame_index:02d}.png"
            pose_path = cloth_dir / f"pose_{frame_index:02d}.npy"
            camera_path = cloth_dir / f"camera_{frame_index:02d}.json"
            Image.fromarray(image, mode="RGB").save(rgb_path)
            Image.fromarray(foreground, mode="L").save(foreground_path)
            Image.fromarray(clothing, mode="L").save(clothing_path)
            np.save(
                pose_path,
                np.array([0.01 * frame_index, 0, 0, 0, 0.02 * frame_index, 0], dtype=np.float32),
            )
            camera = {
                "K": [[32.0, 0.0, 31.5], [0.0, 32.0, 31.5], [0.0, 0.0, 1.0]],
                "w2c": np.eye(4, dtype=np.float32).tolist(),
            }
            camera_path.write_text(json.dumps(camera), encoding="utf-8")
            frame = {
                "frame_id": f"{frame_index:06d}",
                "view_id": f"cam{frame_index:02d}",
                "rgb": str(rgb_path.relative_to(root)),
                "foreground_mask": str(foreground_path.relative_to(root)),
                "pose": str(pose_path.relative_to(root)),
                "camera": str(camera_path.relative_to(root)),
                "Rh": np.eye(3, dtype=np.float32).tolist(),
                "Th": [0.0, 0.0, 0.0],
            }
            if frame_index != 5:
                frame["clothing_mask"] = str(clothing_path.relative_to(root))
            frames.append(frame)
        manifest["clothes"][cloth_name] = {
            "anchor_offset_target": str(target_path.relative_to(root)),
            "frames": frames,
        }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic CanonDressGS episodes")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    path = build_synthetic_image_conditioned_manifest(args.out_dir)
    print(path.resolve())


if __name__ == "__main__":
    main()
