from __future__ import annotations

import json
import copy
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.dressable_dataset import AnchorOffsetDataset, ImageConditionedEpisodeDataset
from scene.dressable_gaussian_model import DressableGaussianModel
from tools.build_synthetic_image_conditioned_manifest import (
    build_synthetic_image_conditioned_manifest,
)
from train_dressable import (
    compute_anchor_training_loss,
    compute_image_conditioned_training_loss,
    create_image_conditioned_components,
    create_training_components,
    evaluate_image_conditioned,
    load_config,
    load_image_training_checkpoint,
    save_image_training_checkpoint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter


class MockRenderBase(nn.Module):
    def __init__(self, anchor_xyz: torch.Tensor) -> None:
        super().__init__()
        num_gaussians = anchor_xyz.shape[0]
        self._xyz = nn.Parameter(anchor_xyz.clone())
        self._scaling = nn.Parameter(torch.zeros(num_gaussians, 3))
        self._opacity = nn.Parameter(torch.zeros(num_gaussians))
        rotation = torch.zeros(num_gaussians, 4)
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._sh0 = nn.Parameter(torch.zeros(num_gaussians, 1, 3))
        self._shN = nn.Parameter(torch.zeros(num_gaussians, 3, 3))
        self.register_buffer("nbr_gs", torch.arange(num_gaussians)[:, None])
        self.register_buffer("nbr_gs_invdist", torch.ones(num_gaussians, 1))
        self.smpl_poses = torch.zeros(6)
        self._dressable_base_fingerprint = "synthetic-render-base"
        self._dressable_checkpoint_path = "mock.pth"

    def render(self, camera, canonical_overrides=None, background=None, **kwargs):
        del background, kwargs
        signal = canonical_overrides["xyz"].mean(dim=0) + 0.2 * canonical_overrides[
            "scaling"
        ].mean(dim=0)
        color = torch.sigmoid(signal)
        alpha_value = torch.sigmoid(canonical_overrides["opacity"].mean())
        rgb = color.view(1, 1, 3).expand(camera["height"], camera["width"], 3)
        alpha = alpha_value.view(1, 1, 1).expand(camera["height"], camera["width"], 1)
        return rgb, alpha, {"mock": True}


def _config(manifest_path: Path, output_dir: Path) -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "canon_dress_gs.yaml")
    config["train"].update(
        {
            "mode": "image_anchor_only",
            "steps": 60,
            "reference_count": 4,
            "batch_size": 1,
            "dual_reference_consistency": True,
            "consistency_every": 1,
            "seed": 42,
            "output_dir": str(output_dir),
            "resume": None,
        }
    )
    config["model"].update(
        {
            "embedding_dim": 16,
            "hidden_dim": 32,
            "num_layers": 3,
            "hyper_hidden_dim": 24,
            "anchor_num_frequencies": 2,
            "anchor_graph_k": 2,
        }
    )
    config["image_conditioning"].update(
        {
            "manifest_path": str(manifest_path),
            "split": "train",
            "feature_dim": 16,
            "embedding_dim": 16,
            "local_feature_dim": 8,
        }
    )
    config["optimizer"].update(
        {
            "encoder_lr": 0.008,
            "aggregator_lr": 0.008,
            "hypernetwork_lr": 0.008,
            "anchor_mlp_lr": 0.008,
            "local_adapter_lr": 0.008,
            "weight_decay": 0.0,
        }
    )
    config["loss"].update(
        {
            "lpips_weight": 0.0,
            "latent_consistency_weight": 0.05,
            "anchor_feature_consistency_weight": 0.05,
            "offset_consistency_weight": 0.05,
        }
    )
    return config


def _activate_path(model) -> None:
    dressable = model.dressable_model
    with torch.no_grad():
        for head in (*dressable.clothing_film_generator.gamma_heads, *dressable.clothing_film_generator.beta_heads):
            head.weight.fill_(0.005)
        dressable.anchor_clothing_mlp.output_layer.weight.normal_(0, 0.01)
        dressable.anchor_clothing_mlp.local_feature_adapter.weight.normal_(0, 0.01)


def _has_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.any(parameter.grad != 0)
        for parameter in parameters
    )


@torch.no_grad()
def _mean_anchor_loss(model, episodes, edges, config) -> float:
    values = []
    for episode in episodes:
        losses, _ = compute_image_conditioned_training_loss(
            model, episode, edges, config, "cpu"
        )
        values.append(losses["anchor"].item())
    return sum(values) / len(values)


def main() -> None:
    print("NOTE: synthetic image-conditioned training test only;")
    print("real MMLPHuman deformation and rendering have not been validated.")
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="canondress_image_training_") as temp_dir:
        root = Path(temp_dir)
        manifest_path = build_synthetic_image_conditioned_manifest(root / "data")
        config = _config(manifest_path, root / "outputs")
        dataset = ImageConditionedEpisodeDataset(manifest_path, "train", 4, seed=42)
        model, optimizer, anchor_features, edges = create_image_conditioned_components(
            dataset, config, device="cpu"
        )
        if model.dressable_model.offset_mode != "anchor_film":
            raise AssertionError("image-conditioned model construction failed")
        print("model construction test: PASS")
        identity_adapter = MMLPHumanAnchorDeformationAdapter.identity(dataset.anchor_xyz)
        if not torch.equal(
            identity_adapter.deform(dataset.anchor_xyz, torch.zeros(6), 0),
            dataset.anchor_xyz,
        ):
            raise AssertionError("explicit identity deformation adapter failed")
        unsafe_adapter = MMLPHumanAnchorDeformationAdapter(None, dataset.anchor_xyz)
        try:
            unsafe_adapter.deform(dataset.anchor_xyz, torch.zeros(6), 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unverified real deformation silently used identity")
        print("deformation adapter safety test: PASS")
        expected_groups = {
            "encoder",
            "aggregator",
            "hypernetwork",
            "anchor_mlp",
            "local_adapter",
        }
        if {group["name"] for group in optimizer.param_groups} != expected_groups:
            raise AssertionError("optimizer parameter groups are incomplete")
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if len(optimizer_ids) != sum(len(group["params"]) for group in optimizer.param_groups):
            raise AssertionError("optimizer contains duplicate parameters")
        print("optimizer parameter group test: PASS")
        embedding_ids = {
            id(parameter) for parameter in model.dressable_model.clothing_embedding.parameters()
        }
        if optimizer_ids & embedding_ids:
            raise AssertionError("cloth-ID embedding leaked into image optimizer")
        print("cloth id embedding exclusion test: PASS")
        if any(parameter.requires_grad for parameter in model.dressable_model.base_model.parameters()):
            raise AssertionError("image-conditioned base is not frozen")
        print("base frozen test: PASS")

        _activate_path(model)
        primary = dataset.sample_episode(0, sampling_salt=1, deterministic=True)
        secondary = dataset.sample_episode(0, sampling_salt=99, deterministic=True)
        losses, outputs = compute_image_conditioned_training_loss(
            model,
            primary,
            edges,
            config,
            "cpu",
            second_episode=secondary,
        )
        if outputs["primary"]["anchor_offsets"]["delta_xyz"].shape != (8, 3):
            raise AssertionError("image-anchor forward returned invalid shape")
        print("image anchor batch forward test: PASS")
        if losses["anchor_available"].item() != 1 or losses["anchor"].item() < 0:
            raise AssertionError("anchor teacher supervision was not enabled")
        no_teacher = copy.deepcopy(primary)
        no_teacher["canonical_anchors"] = dataset.anchor_xyz
        no_teacher["anchor_offset_target"] = None
        no_teacher_losses, _ = compute_image_conditioned_training_loss(
            model, no_teacher, edges, config, "cpu"
        )
        if no_teacher_losses["anchor_available"].item() != 0 or no_teacher_losses[
            "anchor"
        ].item() != 0:
            raise AssertionError("missing anchor teacher did not disable anchor loss")
        print("anchor teacher loss test: PASS")
        if losses["latent_consistency"].item() < 0:
            raise AssertionError("latent consistency is invalid")
        print("dual reference latent consistency test: PASS")
        if losses["anchor_feature_consistency"].item() < 0:
            raise AssertionError("anchor feature consistency is invalid")
        print("anchor feature consistency test: PASS")
        if losses["offset_consistency"].item() < 0:
            raise AssertionError("offset consistency is invalid")
        print("offset consistency test: PASS")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if not _has_gradient(model.clothing_observation_encoder.parameters()):
            raise AssertionError("encoder did not receive gradients")
        print("encoder backward test: PASS")
        if not _has_gradient(model.multiview_aggregator.parameters()):
            raise AssertionError("aggregator did not receive gradients")
        print("aggregator backward test: PASS")
        if not _has_gradient(model.dressable_model.clothing_film_generator.parameters()):
            raise AssertionError("FiLM generator did not receive gradients")
        print("film backward test: PASS")
        if not _has_gradient(model.dressable_model.anchor_clothing_mlp.parameters()):
            raise AssertionError("Anchor MLP did not receive gradients")
        print("anchor mlp backward test: PASS")

        fit_episodes = [
            dataset.sample_episode(0, deterministic=True),
            dataset.sample_episode(6, deterministic=True),
        ]
        initial_loss = _mean_anchor_loss(model, fit_episodes, edges, config)
        for step in range(60):
            episode = fit_episodes[step % 2]
            optimizer.zero_grad(set_to_none=True)
            step_losses, _ = compute_image_conditioned_training_loss(
                model, episode, edges, config, "cpu"
            )
            step_losses["total"].backward()
            optimizer.step()
        final_loss = _mean_anchor_loss(model, fit_episodes, edges, config)
        print(f"initial image anchor loss: {initial_loss:.8f}")
        print(f"final image anchor loss: {final_loss:.8f}")
        if not final_loss < initial_loss:
            raise AssertionError("image-conditioned fitting did not reduce anchor loss")
        print("training loss reduction test: PASS")
        blue = model.forward_episode(fit_episodes[0])["anchor_offsets"]["delta_xyz"]
        red = model.forward_episode(fit_episodes[1])["anchor_offsets"]["delta_xyz"]
        if torch.allclose(blue, red):
            raise AssertionError("different clothing produces identical offsets")
        print("different clothing output separation test: PASS")
        same_other = dataset.sample_episode(0, sampling_salt=77, deterministic=True)
        same_output = model.forward_episode(same_other)["anchor_offsets"]["delta_xyz"]
        if not torch.mean((blue - same_output).square()) < torch.mean((blue - red).square()):
            raise AssertionError("same-clothing references are not more consistent")
        print("same clothing reference consistency test: PASS")

        checkpoint_path = root / "image_checkpoint.pth"
        save_image_training_checkpoint(
            checkpoint_path, model, optimizer, 60, config, dataset
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != 2 or checkpoint["manifest_fingerprint"] != dataset.manifest_fingerprint:
            raise AssertionError("image checkpoint metadata is incomplete")
        print("image checkpoint save test: PASS")
        restored, restored_optimizer, _, _ = create_image_conditioned_components(
            dataset, config, device="cpu"
        )
        restored_step = load_image_training_checkpoint(
            checkpoint_path, restored, restored_optimizer, config, dataset
        )
        if restored_step != 60:
            raise AssertionError("image checkpoint resume restored wrong step")
        print("image checkpoint resume test: PASS")
        original_fingerprint = dataset.manifest_fingerprint
        dataset.manifest_fingerprint = "mismatch"
        try:
            load_image_training_checkpoint(
                checkpoint_path, restored, restored_optimizer, config, dataset
            )
        except ValueError as error:
            if "manifest_fingerprint" not in str(error):
                raise
        else:
            raise AssertionError("manifest fingerprint mismatch was accepted")
        dataset.manifest_fingerprint = original_fingerprint
        print("manifest fingerprint mismatch test: PASS")
        evaluate_image_conditioned(model, dataset, 60, root / "eval")
        prediction_root = root / "eval" / "predictions" / "step_000060"
        if not all(
            (prediction_root / cloth / "predicted_anchor_offsets.pt").is_file()
            for cloth in ("cloth_blue", "cloth_red")
        ):
            raise AssertionError("image-conditioned evaluation export is incomplete")
        print("image evaluation export test: PASS")

        mock_base = MockRenderBase(dataset.anchor_xyz)
        render_model, render_optimizer, _, render_edges = create_image_conditioned_components(
            dataset, config, mock_base, "cpu"
        )
        _activate_path(render_model)
        render_losses, render_outputs = compute_image_conditioned_training_loss(
            render_model,
            primary,
            render_edges,
            config,
            "cpu",
            render_target=True,
            deformation_fn=lambda anchors, pose, index: anchors,
        )
        pred_rgb, pred_alpha = render_outputs["primary"]["render"][:2]
        if pred_rgb.shape != (64, 64, 3) or pred_alpha.shape != (64, 64, 1):
            raise AssertionError("mock image render returned invalid shape")
        print("mock image render batch test: PASS")
        render_optimizer.zero_grad(set_to_none=True)
        render_losses["total"].backward()
        if not _has_gradient(render_model.clothing_observation_encoder.parameters()):
            raise AssertionError("render loss did not reach image model")
        print("render loss gradient test: PASS")
        evaluate_image_conditioned(
            render_model,
            dataset,
            1,
            root / "render_eval",
            render_target=True,
            deformation_fn=lambda anchors, pose, index: anchors,
        )
        if not list((root / "render_eval" / "vis").rglob("rgb_difference.png")):
            raise AssertionError("render evaluation visualization was not exported")
        print("render visualization export test: PASS")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_paths = [
            manifest_path.parent / manifest["clothes"][name]["anchor_offset_target"]
            for name in manifest["splits"]["train"]
        ]
        anchor_dataset = AnchorOffsetDataset(target_paths)
        old_model, _, _, old_edges = create_training_components(anchor_dataset, config, "cpu")
        old_loss = compute_anchor_training_loss(
            old_model, anchor_dataset[0], old_edges, config["loss"], "cpu"
        )
        if not torch.isfinite(old_loss["total"]):
            raise AssertionError("old anchor-only mode regressed")
        print("anchor only old mode regression test: PASS")
        old_render = DressableGaussianModel(MockRenderBase(dataset.anchor_xyz))
        old_render.initialize_film_clothing_generator(
            4,
            anchor_features,
            torch.arange(8)[:, None],
            torch.ones(8, 1),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=3,
            hyper_hidden_dim=24,
        )
        rgb, alpha, _ = old_render.render(
            {"height": 8, "width": 8}, 0, background=torch.ones(3)
        )
        if rgb.shape != (8, 8, 3) or alpha.shape != (8, 8, 1):
            raise AssertionError("old rendering path regressed")
        print("rendering mock old mode regression test: PASS")
        if not all(torch.isfinite(value) for value in (*losses.values(), *render_losses.values())):
            raise AssertionError("training tests produced non-finite losses")
        print("all image losses finite test: PASS")
    print("NOTE: synthetic image-conditioned training test only;")
    print("real MMLPHuman deformation and rendering have not been validated.")


if __name__ == "__main__":
    main()
