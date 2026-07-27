from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training
from scene.anchor_image_projector import AnchorImageProjector
from scene.dressable_dataset import (
    ImageConditionedEpisodeDataset,
    image_conditioned_episode_collate,
)
from tools.build_synthetic_image_conditioned_manifest import (
    build_synthetic_image_conditioned_manifest,
)
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


class StatefulMockBase(nn.Module):
    def __init__(self, anchors: torch.Tensor) -> None:
        super().__init__()
        num_gaussians = anchors.shape[0]
        self._xyz = nn.Parameter(anchors.clone())
        self._scaling = nn.Parameter(torch.zeros(num_gaussians, 3))
        self._opacity = nn.Parameter(torch.zeros(num_gaussians))
        rotation = torch.zeros(num_gaussians, 4)
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._sh0 = nn.Parameter(torch.zeros(num_gaussians, 1, 3))
        self._shN = nn.Parameter(torch.zeros(num_gaussians, 3, 3))
        self.register_buffer("xyz_vt", anchors.clone())
        self.register_buffer("nbr_gs", torch.arange(num_gaussians)[:, None])
        self.register_buffer("nbr_gs_invdist", torch.ones(num_gaussians, 1))
        self._smpl_poses = torch.zeros(165)
        self.smpl_poses_cuda = self._smpl_poses.clone()
        self._Rh = torch.eye(3)
        self._Th = torch.zeros(3)
        self.cache_dict = {"sentinel": object()}
        self._dressable_base_fingerprint = "synthetic-stateful-base"
        self.last_render_state = None

    @property
    def smpl_poses(self):
        return self._smpl_poses

    @smpl_poses.setter
    def smpl_poses(self, value):
        self.cache_dict = {}
        self._smpl_poses = value.detach().cpu()
        self.smpl_poses_cuda = value.detach().clone()

    @property
    def Rh(self):
        return self._Rh

    @Rh.setter
    def Rh(self, value):
        self._Rh = value.detach().clone()

    @property
    def Th(self):
        return self._Th

    @Th.setter
    def Th(self, value):
        self._Th = value.detach().clone()

    def render(self, camera, canonical_overrides=None, background=None, **kwargs):
        del background, kwargs
        self.last_render_state = (
            self._smpl_poses.detach().clone(),
            self._Rh.detach().clone(),
            self._Th.detach().clone(),
        )
        signal = canonical_overrides["xyz"].mean(dim=0)
        rgb = torch.sigmoid(signal).view(1, 1, 3).expand(
            camera["height"], camera["width"], 3
        )
        alpha = torch.sigmoid(canonical_overrides["opacity"].mean()).view(1, 1, 1)
        alpha = alpha.expand(camera["height"], camera["width"], 1)
        return rgb, alpha, {"mock": True}


class RecordingAdapter:
    def __init__(self) -> None:
        self.calls = []

    def deform_anchors(self, anchors, pose, Rh, Th):
        self.calls.append((pose.detach().clone(), Rh.detach().clone(), Th.detach().clone()))
        return anchors + Th


def _write_manifest(root: Path, name: str, manifest: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _anchor_xyz() -> torch.Tensor:
    return torch.stack(
        (torch.linspace(-0.25, 0.25, 8), torch.zeros(8), torch.full((8,), 2.0)),
        dim=1,
    )


def _activate_image_path(model) -> None:
    dressable = model.dressable_model
    with torch.no_grad():
        for head in (
            *dressable.clothing_film_generator.gamma_heads,
            *dressable.clothing_film_generator.beta_heads,
        ):
            head.weight.fill_(0.005)
        dressable.anchor_clothing_mlp.output_layer.weight.normal_(0, 0.01)
        dressable.anchor_clothing_mlp.local_feature_adapter.weight.normal_(0, 0.01)


def _has_nonzero_finite_grad(parameters) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.any(parameter.grad != 0)
        for parameter in parameters
    )


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="canondress_gate2_interfaces_") as temp_dir:
        root = Path(temp_dir)
        manifest_path = build_synthetic_image_conditioned_manifest(root / "data")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        inline_dataset = ImageConditionedEpisodeDataset(manifest_path, "train", 2)
        inline_episode = inline_dataset[0]
        if inline_episode["reference_Rh"].shape != (2, 3, 3) or inline_episode[
            "reference_Th"
        ].shape != (2, 3):
            raise AssertionError("inline Rh/Th were not returned with K")
        print("manifest inline Rh/Th test: PASS")

        npy_manifest = copy.deepcopy(manifest)
        rh_path = root / "data" / "rh_test.npy"
        th_path = root / "data" / "th_test.npy"
        np.save(rh_path, np.eye(3, dtype=np.float32))
        np.save(th_path, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        frame = npy_manifest["clothes"]["cloth_red"]["frames"][0]
        frame["Rh"] = rh_path.name
        frame["Th"] = th_path.name
        npy_dataset = ImageConditionedEpisodeDataset(
            _write_manifest(root / "data", "manifest_npy.json", npy_manifest),
            "train",
            2,
        )
        loaded_frame = npy_dataset._load_frame(
            npy_dataset._resolved_clothes["cloth_red"]["frames"][0]
        )
        if not torch.allclose(loaded_frame["Th"], torch.tensor([0.1, 0.2, 0.3])):
            raise AssertionError("npy Rh/Th contents were not loaded")
        print("manifest npy Rh/Th test: PASS")

        missing = copy.deepcopy(manifest)
        missing_frame = missing["clothes"]["cloth_red"]["frames"][0]
        missing_frame.pop("Rh")
        missing_frame.pop("Th")
        missing_path = _write_manifest(root / "data", "manifest_missing_rt.json", missing)
        try:
            ImageConditionedEpisodeDataset(missing_path, "train", 2)
        except ValueError:
            pass
        else:
            raise AssertionError("strict dataset accepted missing Rh/Th")
        ImageConditionedEpisodeDataset(
            missing_path, "train", 2, allow_missing_rh_th=True
        )
        one_missing = copy.deepcopy(manifest)
        one_missing["clothes"]["cloth_red"]["frames"][0].pop("Th")
        try:
            ImageConditionedEpisodeDataset(
                _write_manifest(root / "data", "manifest_one_rt.json", one_missing),
                "train",
                2,
                allow_missing_rh_th=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("dataset accepted only one of Rh/Th")
        print("missing Rh/Th strict rejection test: PASS")

        for name, invalid_value in (
            ("shape", [[1.0, 0.0], [0.0, 1.0]]),
            ("nan", [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]),
        ):
            invalid = copy.deepcopy(manifest)
            invalid["clothes"]["cloth_red"]["frames"][0]["Rh"] = invalid_value
            try:
                ImageConditionedEpisodeDataset(
                    _write_manifest(root / "data", f"manifest_{name}.json", invalid),
                    "train",
                    2,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"dataset accepted invalid Rh {name}")
        print("Rh/Th shape and finite validation test: PASS")

        if image_conditioned_episode_collate([inline_episode]) is not inline_episode:
            raise AssertionError("collate changed Rh/Th episode")
        print("Rh/Th collate preservation test: PASS")

        no_teacher = copy.deepcopy(manifest)
        for cloth in no_teacher["clothes"].values():
            cloth.pop("anchor_offset_target", None)
        no_teacher_path = _write_manifest(
            root / "data", "manifest_no_teacher.json", no_teacher
        )
        no_teacher_dataset = ImageConditionedEpisodeDataset(
            no_teacher_path, "train", 2
        )
        anchors = _anchor_xyz()
        large_base = type("LargeBase", (), {"xyz_vt": torch.zeros(10000, 3)})()
        resolved = training.resolve_image_conditioned_canonical_anchors(
            no_teacher_dataset, large_base, "cpu", require_real_base=True
        )
        if resolved.shape != (10000, 3):
            raise AssertionError("real base anchors were not selected without teacher")
        print("real base anchor priority test: PASS")

        mismatch_base = type("MismatchBase", (), {"xyz_vt": anchors + 0.01})()
        try:
            training.resolve_image_conditioned_canonical_anchors(
                inline_dataset, mismatch_base, "cpu"
            )
        except ValueError:
            pass
        else:
            raise AssertionError("teacher/base anchor mismatch was accepted")
        print("teacher anchor mismatch rejection test: PASS")

        state_base = StatefulMockBase(anchors)
        original_cache = state_base.cache_dict
        original_cache_contents = dict(original_cache)
        original_state = {
            name: getattr(state_base, name)
            for name in ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
        }
        pose = torch.linspace(0, 0.1, 165)
        Rh = torch.eye(3)
        Th = torch.tensor([0.2, 0.0, 0.0])
        with mmlphuman_state_transaction(state_base, pose, Rh, Th):
            if not torch.equal(state_base._smpl_poses, pose) or state_base.cache_dict:
                raise AssertionError("transaction did not install state/clear cache")
        if state_base.cache_dict is not original_cache or state_base.cache_dict != original_cache_contents:
            raise AssertionError("transaction did not restore cache object/content")
        if any(getattr(state_base, name) is not value for name, value in original_state.items()):
            raise AssertionError("transaction did not restore state objects")
        print("state transaction normal restoration test: PASS")

        try:
            with mmlphuman_state_transaction(state_base, pose, Rh, Th):
                raise RuntimeError("intentional")
        except RuntimeError as error:
            if str(error) != "intentional":
                raise
        if state_base.cache_dict is not original_cache or any(
            getattr(state_base, name) is not value for name, value in original_state.items()
        ):
            raise AssertionError("exception path polluted base state")
        print("state transaction exception restoration test: PASS")
        print("cache object and content restoration test: PASS")

        real_episode = copy.deepcopy(no_teacher_dataset[0])
        real_episode["reference_poses"] = torch.zeros(2, 165)
        real_episode["reference_Th"][0] = torch.tensor([0.0, 0.0, 0.0])
        real_episode["reference_Th"][1] = torch.tensor([0.1, 0.0, 0.0])
        adapter = RecordingAdapter()
        closure = training.build_episode_deformation_fn(adapter, real_episode)
        closure(anchors, real_episode["reference_poses"][0], 0)
        closure(anchors, real_episode["reference_poses"][1], 1)
        if not torch.equal(adapter.calls[1][2], real_episode["reference_Th"][1]):
            raise AssertionError("deformation closure used the wrong view transform")
        print("per-view Rh/Th deformation closure test: PASS")

        observed_translations = []
        original_depth_renderer = training.render_mmlphuman_expected_depth

        def fake_depth_renderer(base_model, camera, background, **kwargs):
            del background, kwargs
            observed_translations.append(base_model._Th.detach().clone())
            height, width = camera["height"], camera["width"]
            return {
                "depth": torch.full((height, width), 2.0),
                "alpha": torch.ones(height, width),
                "render_mode": "ED",
            }

        training.render_mmlphuman_expected_depth = fake_depth_renderer
        try:
            geometry = training.prepare_real_reference_geometry(
                state_base, adapter, real_episode, torch.ones(3)
            )
        finally:
            training.render_mmlphuman_expected_depth = original_depth_renderer
        if geometry["surface_depth_maps"].shape != (2, 1, 64, 64) or geometry[
            "surface_alpha_maps"
        ].shape != (2, 1, 64, 64):
            raise AssertionError("reference ED stack has the wrong shape")
        if not torch.equal(observed_translations[1], real_episode["reference_Th"][1]):
            raise AssertionError("reference ED renderer used the wrong view state")
        print("reference K-view state selection test: PASS")
        print("reference ED depth stack shape test: PASS")

        projector = AnchorImageProjector()
        try:
            projector.project_and_sample(
                anchors,
                torch.ones(2, 4, 8, 8),
                torch.zeros(2, 6),
                real_episode["reference_cameras"],
                64,
                64,
                deformation_fn=lambda value, pose_value, index: value,
                require_depth_visibility=True,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("projector silently accepted missing required depth")
        print("required depth fail-fast test: PASS")

        config = training.load_config(PROJECT_ROOT / "configs" / "canon_dress_gs.yaml")
        config["model"].update(
            {
                "enable_delta_xyz": True,
                "enable_delta_scaling": False,
                "enable_delta_opacity": False,
            }
        )
        config["image_conditioning"]["allow_identity_deformation"] = True
        model, optimizer, _, edges = training.create_image_conditioned_components(
            no_teacher_dataset, config, state_base, "cpu"
        )
        model_episode = copy.deepcopy(no_teacher_dataset[0])
        num_views = model_episode["reference_images"].shape[0]
        depth_maps = torch.full((num_views, 1, 64, 64), 2.015)
        alpha_maps = torch.ones_like(depth_maps)
        output = model.forward_episode(
            model_episode,
            deformation_fn=lambda value, pose_value, index: value,
            surface_depth_maps=depth_maps,
            surface_alpha_maps=alpha_maps,
            depth_abs_tolerance=0.02,
            depth_rel_tolerance=0.0,
            depth_alpha_threshold=1e-4,
            require_depth_visibility=True,
        )
        if "depth_visible_mask" not in output["projection"]:
            raise AssertionError("model dropped depth parameters before the projector")
        if not torch.any(output["projection"]["depth_visible_mask"]):
            raise AssertionError("depth tolerance was not applied by the projector")
        print("depth parameter propagation test: PASS")
        if output["dressed_canonical_params"]["xyz"].shape != anchors.shape:
            raise AssertionError("teacherless forward did not use model canonical anchors")
        print("no-teacher image forward test: PASS")

        _activate_image_path(model)
        optimizer.zero_grad(set_to_none=True)
        model_episode["target_pose"] = torch.zeros(165)
        original_cache = state_base.cache_dict
        original_state = {
            name: getattr(state_base, name)
            for name in ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
        }
        render_output = model.forward_episode(
            model_episode,
            deformation_fn=lambda value, pose_value, index: value,
            render_target=True,
            render_kwargs={"background": torch.ones(3)},
            require_mmlphuman_state_transaction=True,
        )
        if state_base.cache_dict is not original_cache or any(
            getattr(state_base, name) is not value for name, value in original_state.items()
        ):
            raise AssertionError("target render polluted pose/Rh/Th/cache state")
        installed_pose, installed_Rh, installed_Th = state_base.last_render_state
        if not torch.equal(installed_pose, model_episode["target_pose"]) or not torch.equal(
            installed_Rh, model_episode["target_Rh"]
        ) or not torch.equal(installed_Th, model_episode["target_Th"]):
            raise AssertionError("target render did not install pose/Rh/Th")
        print("target pose Rh Th transaction test: PASS")

        reference_gate = torch.zeros(anchors.shape[0])
        reference_gate[: anchors.shape[0] // 2] = 1
        gated_output = model.forward_episode(
            model_episode,
            deformation_fn=lambda value, pose_value, index: value,
            reference_only_gate=reference_gate,
        )
        if gated_output["anchor_offsets"]["delta_xyz"][anchors.shape[0] // 2 :].abs().max() != 0:
            raise AssertionError("reference-only gate did not zero excluded anchors")
        print("reference-only offset gate test: PASS")

        anchor_offsets = render_output["anchor_offsets"]
        gaussian_offsets = render_output["gaussian_offsets"]
        if anchor_offsets["delta_scaling"].abs().max() != 0 or anchor_offsets[
            "delta_opacity"
        ].abs().max() != 0:
            raise AssertionError("disabled anchor channels are not exactly zero")
        if gaussian_offsets["delta_scaling"].abs().max() != 0 or gaussian_offsets[
            "delta_opacity"
        ].abs().max() != 0:
            raise AssertionError("disabled Gaussian channels are not exactly zero")
        print("xyz-only offset channel gate test: PASS")

        pred_rgb, pred_alpha = render_output["render"][:2]
        loss = pred_rgb.mean() + pred_alpha.mean() + anchor_offsets["delta_xyz"].square().mean()
        loss.backward()
        if any(parameter.grad is not None for parameter in state_base.parameters()):
            raise AssertionError("frozen base received gradients")
        print("frozen base no-gradient test: PASS")
        trainable_groups = (
            model.clothing_observation_encoder.parameters(),
            model.multiview_aggregator.parameters(),
            model.dressable_model.clothing_film_generator.parameters(),
            model.dressable_model.anchor_clothing_mlp.parameters(),
        )
        if not all(_has_nonzero_finite_grad(group) for group in trainable_groups):
            raise AssertionError("image-conditioned trainable groups lack finite gradients")
        print("image-conditioned trainable gradient test: PASS")

        global_only = model.encode_global_clothing(
            model_episode["reference_images"],
            model_episode["reference_cloth_masks"],
            model_episode["reference_valid_mask"],
        )
        if global_only["global_clothing_embedding"].shape[0] != 1:
            raise AssertionError("global-only compatibility path regressed")
        print("legacy synthetic global-only compatibility test: PASS")

    print("Gate 2 image-conditioned interface checks: 22 PASS")


if __name__ == "__main__":
    main()
