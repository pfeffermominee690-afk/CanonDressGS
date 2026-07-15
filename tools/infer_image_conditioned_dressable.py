from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import train_dressable as training
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from tools.check_real_image_conditioned_one_batch import save_render_tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_frame(dataset, frame_id: str, view_id: str):
    matches = []
    for cloth in dataset._resolved_clothes.values():
        matches.extend(
            frame for frame in cloth["frames"]
            if str(frame["frame_id"]) == frame_id and str(frame["view_id"]) == view_id
        )
    if len(matches) != 1:
        raise ValueError(f"expected one frame {frame_id}/{view_id}, found {len(matches)}")
    return matches[0]


def _target_geometry(dataset, frame, height: int, width: int):
    pose = torch.from_numpy(np.asarray(np.load(frame["pose"], allow_pickle=False))).float().reshape(-1)
    with frame["camera"].open("r", encoding="utf-8") as handle:
        camera = json.load(handle)
    rh, th, fallback = dataset._load_rh_th(frame)
    if fallback:
        raise ValueError("independent real inference forbids Rh/Th fallback")
    return {
        "target_rgb": torch.zeros(3, height, width),
        "target_pose": pose,
        "target_Rh": rh,
        "target_Th": th,
        "target_camera": camera,
        "target_frame_id": str(frame["frame_id"]),
        "target_view_id": str(frame["view_id"]),
    }


def build_inference_episode(dataset, references, target):
    target_key = tuple(target)
    if target_key in {tuple(item) for item in references}:
        raise ValueError("target must not appear in references")
    loaded = [dataset._load_frame(_find_frame(dataset, *item)) for item in references]
    height, width = loaded[0]["rgb"].shape[-2:]
    if any(tuple(item["rgb"].shape[-2:]) != (height, width) for item in loaded):
        raise ValueError("reference resolutions do not match")
    episode = {
        "identity_id": dataset.identity_id,
        "cloth_name": next(iter(dataset._resolved_clothes)),
        "cloth_id": torch.tensor(0, dtype=torch.long),
        "episode_index": 0,
        "reference_images": torch.stack([item["rgb"] for item in loaded]),
        "reference_cloth_masks": torch.stack([item["clothing_mask"] for item in loaded]),
        "reference_foreground_masks": torch.stack([item["foreground_mask"] for item in loaded]),
        "reference_poses": torch.stack([item["pose"] for item in loaded]),
        "reference_Rh": torch.stack([item["Rh"] for item in loaded]),
        "reference_Th": torch.stack([item["Th"] for item in loaded]),
        "reference_cameras": [item["camera"] for item in loaded],
        "reference_frame_ids": [item["frame_id"] for item in loaded],
        "reference_view_ids": [item["view_id"] for item in loaded],
        "reference_valid_mask": torch.ones(len(loaded)),
    }
    episode.update(_target_geometry(dataset, _find_frame(dataset, *target), height, width))
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent image-conditioned dressable inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--lbs-grid", required=True)
    parser.add_argument("--reference-region", required=True)
    parser.add_argument("--reference", action="append", nargs=2, metavar=("FRAME", "CAMERA"), required=True)
    parser.add_argument("--target", nargs=2, metavar=("FRAME", "CAMERA"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(saved.get("step", -1)) != 100:
        raise ValueError("independent inference requires checkpoint step 100")
    config = saved["config"]
    config["image_conditioning"]["manifest_path"] = str(Path(args.manifest).resolve())
    config["base"]["model_dir"] = str(Path(args.base_model_dir).resolve())
    config["base"]["checkpoint_path"] = str(Path(args.base_checkpoint).resolve())
    config["base"]["lbs_grid_path"] = str(Path(args.lbs_grid).resolve())
    config["image_conditioning"]["reference_region_path"] = str(Path(args.reference_region).resolve())
    device = torch.device(args.device)
    dataset = ImageConditionedEpisodeDataset(
        args.manifest, split=config["image_conditioning"]["split"],
        reference_count=len(args.reference), seed=int(config["train"].get("seed", 0)),
        allow_clothing_mask_fallback=False, allow_missing_rh_th=False,
    )
    base = training.load_frozen_mmlphuman_base(args.base_model_dir, args.base_checkpoint, device)
    model, optimizer, _, _ = training.create_image_conditioned_components(dataset, config, base, device)
    fingerprint = getattr(base, "_dressable_base_fingerprint", None)
    restored = training.load_image_training_checkpoint(checkpoint_path, model, optimizer, config, dataset, fingerprint)
    region_object = torch.load(args.reference_region, map_location="cpu", weights_only=True)
    score = region_object["cloth_region_weight"].float().reshape(-1)
    gate = (score >= float(config["image_conditioning"]["reference_gate_threshold"])).to(device)
    adapter = training.MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base, canonical_anchors=model.canonical_anchors, lbs_grid_path=args.lbs_grid
    )
    episode = build_inference_episode(dataset, args.reference, args.target)
    episode["reference_only_gate"] = gate
    geometry = training.prepare_real_reference_geometry(
        base, adapter, episode, config.get("render", {}).get("background")
    )
    background = torch.as_tensor(
        config.get("render", {}).get("background", [1, 1, 1]),
        device=device,
        dtype=model.dressable_model.anchor_features.dtype,
    )

    def infer_once():
        model.eval()
        with torch.no_grad():
            return model.forward_episode(
                episode, deformation_fn=geometry["deformation_fn"], render_target=True,
                surface_depth_maps=geometry["surface_depth_maps"],
                surface_alpha_maps=geometry["surface_alpha_maps"],
                require_depth_visibility=True,
                depth_abs_tolerance=float(config["image_conditioning"]["depth_abs_tolerance"]),
                depth_rel_tolerance=float(config["image_conditioning"]["depth_rel_tolerance"]),
                depth_alpha_threshold=float(config["image_conditioning"]["depth_alpha_threshold"]),
                require_mmlphuman_state_transaction=True,
                render_kwargs={"background": background},
            )

    first, second = infer_once(), infer_once()
    tensors = {
        "anchor": first["anchor_offsets"]["delta_xyz"],
        "gaussian": first["gaussian_offsets"]["delta_xyz"],
        "rgb": first["render"][0],
        "alpha": first["render"][1],
    }
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise FloatingPointError("independent inference output contains NaN or Inf")
    repeat = {key: float((value - ({"anchor": second["anchor_offsets"]["delta_xyz"], "gaussian": second["gaussian_offsets"]["delta_xyz"], "rgb": second["render"][0], "alpha": second["render"][1]}[key])).abs().max().item()) for key, value in tensors.items()}
    disabled = {
        key: float(first[group][channel].abs().max().item())
        for group in ("anchor_offsets", "gaussian_offsets")
        for key, channel in ((f"{group}_scaling", "delta_scaling"), (f"{group}_opacity", "delta_opacity"))
    }
    if any(repeat.values()) or any(disabled.values()):
        raise RuntimeError(f"inference determinism/disabled-channel failure: {repeat}, {disabled}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in first["anchor_offsets"].items()}, output / "predicted_anchor_offsets.pt")
    torch.save({"rgb": tensors["rgb"].cpu(), "alpha": tensors["alpha"].cpu()}, output / "inference_render.pt")
    save_render_tensor(output / "step100_rgb.png", tensors["rgb"], 3)
    save_render_tensor(output / "rendered_alpha.png", tensors["alpha"], 1)
    manifest = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": restored, "manifest": str(Path(args.manifest).resolve()),
        "references": args.reference, "target_pose_camera": args.target,
        "target_view_used": False, "target_rgb_read": False, "target_mask_read": False,
        "teacher_read": False, "cloth_id_used_as_condition": False,
        "reference_region": str(Path(args.reference_region).resolve()),
        "reference_region_sha256": _sha256(Path(args.reference_region)),
        "repeat_max_abs_diff": repeat, "disabled_max_abs": disabled,
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
    }
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
