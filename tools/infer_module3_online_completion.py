from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import train_dressable as training
from scene.gaussian_clothing_residuals import CHANNELS
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


class _InferenceModelContract:
    def __init__(self, num_clothes: int):
        self.num_clothes = int(num_clothes)

    @staticmethod
    def get_anchor_xyz_or_none():
        return None


def _num_clothes(state: dict[str, torch.Tensor]) -> int:
    matches = [value for key, value in state.items() if key.endswith("clothing_embedding.embedding.weight")]
    if len(matches) != 1 or matches[0].ndim != 2:
        raise ValueError("checkpoint must contain one clothing embedding table")
    return int(matches[0].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate4-config", required=True)
    parser.add_argument("--module2-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    request = torch.load(args.request, map_location="cpu", weights_only=True)
    forbidden = {
        "teacher", "teacher_gate", "teacher_residuals", "target_rgb",
        "target_foreground_mask", "target_clothing_mask", "cloth_id",
        "reference_only_gate",
    }.intersection(request)
    if forbidden:
        raise ValueError(f"forbidden independent inference fields: {sorted(forbidden)}")
    required = {
        "reference_images", "reference_cloth_masks", "reference_foreground_masks",
        "reference_poses", "reference_Rh", "reference_Th", "reference_cameras",
        "reference_valid_mask", "target_pose", "target_Rh", "target_Th",
        "target_camera", "target_height", "target_width",
    }
    if set(request) != required:
        raise ValueError(f"independent inference request fields mismatch: {sorted(set(request) ^ required)}")

    c4 = training.load_config(args.gate4_config)
    c2 = training.load_config(args.module2_config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint["model"]
    base = training.load_frozen_mmlphuman_base(
        c4["base"]["model_dir"], c4["base"]["checkpoint_path"], device
    )
    contract = _InferenceModelContract(_num_clothes(state))
    model, _, _, _ = training.create_image_conditioned_components(contract, c4, base, device)
    model.dressable_model.configure_six_channel_decoder(c2["dressable_channels"])
    metadata = checkpoint["metadata"]
    graph_indices = checkpoint["graph_indices"].to(device)
    graph_weights = checkpoint["graph_weights"].to(device=device, dtype=base._xyz.dtype)
    model.initialize_online_completion(
        graph_indices, graph_weights,
        int(metadata["feature_completion_hidden_dim"]),
        int(metadata["feature_completion_blocks"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    episode = {
        key: value for key, value in request.items()
        if key.startswith("reference_")
    }
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base, canonical_anchors=model.canonical_anchors,
        lbs_grid_path=c4["base"]["lbs_grid_path"],
    )
    background = torch.tensor(c4["render"]["background"], device=device, dtype=base._xyz.dtype)
    geometry = training.prepare_real_reference_geometry(base, adapter, episode, background)
    online_inputs = {
        "reference_images": request["reference_images"].to(device),
        "reference_cloth_masks": request["reference_cloth_masks"].to(device),
        "reference_foreground_masks": request["reference_foreground_masks"].to(device),
        "reference_poses": request["reference_poses"].to(device),
        "reference_cameras": request["reference_cameras"],
        "reference_valid_mask": request["reference_valid_mask"].to(device),
        "deformation_fn": training.build_episode_deformation_fn(adapter, episode),
        "surface_depth_maps": geometry["surface_depth_maps"],
        "surface_alpha_maps": geometry["surface_alpha_maps"],
        "require_depth_visibility": True,
    }
    with torch.no_grad():
        result = model.compute_online_six_channel_residuals(**online_inputs)
        overrides = model.dressable_model.compose_canonical_gaussian_overrides(
            result["gaussian_residuals"], CHANNELS
        )
        camera = build_mmlphuman_camera(
            request["target_camera"], int(request["target_height"]),
            int(request["target_width"]), device,
        )
        original_degree = int(base.sh_degree)
        try:
            base.sh_degree = int(metadata["module2_decoder_metadata"]["effective_sh_degree"])
            with mmlphuman_state_transaction(
                base, request["target_pose"].to(device), request["target_Rh"].to(device),
                request["target_Th"].to(device),
            ):
                rgb, alpha, _ = base.render(camera, background=background, canonical_overrides=overrides.as_dict())
        finally:
            base.sh_degree = original_degree
    output = {
        "completed_features": result["completion"].completed_anchor_features.cpu(),
        "geometry_gate": result["completion"].geometry_gate.cpu(),
        "appearance_gate": result["completion"].appearance_gate.cpu(),
        "confidence": result["completion"].confidence.cpu(),
        "RGB": rgb.cpu(), "alpha": alpha.cpu(),
    }
    output.update({f"anchor.{name}": getattr(result["gated_anchor_residuals"], name).cpu() for name in CHANNELS})
    output.update({f"Gaussian.{name}": getattr(result["gaussian_residuals"], name).cpu() for name in CHANNELS})
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
