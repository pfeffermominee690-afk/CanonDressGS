from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from scene.anchor_image_projector import AnchorImageProjector
from scene.clothing_observation_encoder import ClothingObservationEncoder
from scene.dressable_gaussian_model import DressableGaussianModel
from scene.gaussian_clothing_residuals import apply_protected_full_residual_guard
from scene.multiview_clothing_aggregator import MultiViewClothingAggregator
from scene.canonical_clothing_completion import CanonicalClothingCompleter
from utils.anchor_graph_utils import validate_anchor_graph
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


class ImageConditionedDressableModel(nn.Module):
    """Compose global and optional anchor-local image clothing conditions."""

    def __init__(
        self,
        dressable_model: DressableGaussianModel,
        clothing_observation_encoder: ClothingObservationEncoder | None = None,
        anchor_image_projector: AnchorImageProjector | None = None,
        multiview_aggregator: MultiViewClothingAggregator | None = None,
        canonical_anchors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dressable_model, DressableGaussianModel):
            raise TypeError("dressable_model must be a DressableGaussianModel")
        if dressable_model.clothing_film_generator is None:
            raise RuntimeError(
                "dressable_model must initialize its FiLM clothing generator first"
            )
        if dressable_model.offset_mode != "anchor_film":
            raise ValueError("image-conditioned dressing requires offset_mode='anchor_film'")
        expected_dim = dressable_model.clothing_film_generator.clothing_dim
        device = dressable_model.anchor_features.device
        dtype = dressable_model.anchor_features.dtype
        if clothing_observation_encoder is None:
            clothing_observation_encoder = ClothingObservationEncoder(
                embedding_dim=expected_dim
            ).to(device=device, dtype=dtype)
        if not isinstance(clothing_observation_encoder, ClothingObservationEncoder):
            raise TypeError(
                "clothing_observation_encoder must be a ClothingObservationEncoder"
            )
        if clothing_observation_encoder.embedding_dim != expected_dim:
            raise ValueError(
                "encoder embedding_dim must match the FiLM clothing dimension: "
                f"{clothing_observation_encoder.embedding_dim} != {expected_dim}"
            )
        if anchor_image_projector is None:
            anchor_image_projector = AnchorImageProjector()
        if not isinstance(anchor_image_projector, AnchorImageProjector):
            raise TypeError("anchor_image_projector must be an AnchorImageProjector")
        if multiview_aggregator is None:
            multiview_aggregator = MultiViewClothingAggregator(
                input_dim=clothing_observation_encoder.feature_dim,
                output_dim=clothing_observation_encoder.feature_dim,
            ).to(device=device, dtype=dtype)
        if not isinstance(multiview_aggregator, MultiViewClothingAggregator):
            raise TypeError(
                "multiview_aggregator must be a MultiViewClothingAggregator"
            )
        if multiview_aggregator.input_dim != clothing_observation_encoder.feature_dim:
            raise ValueError(
                "aggregator input_dim must match encoder feature_dim: "
                f"{multiview_aggregator.input_dim} != "
                f"{clothing_observation_encoder.feature_dim}"
            )
        dressable_model.anchor_clothing_mlp.initialize_local_feature_adapter(
            multiview_aggregator.output_dim
        )
        self.dressable_model = dressable_model
        self.clothing_observation_encoder = clothing_observation_encoder
        self.anchor_image_projector = anchor_image_projector
        self.multiview_aggregator = multiview_aggregator
        if canonical_anchors is None:
            anchors = torch.empty(0, 3, device=device, dtype=dtype)
        else:
            if (
                not isinstance(canonical_anchors, torch.Tensor)
                or canonical_anchors.ndim != 2
                or canonical_anchors.shape[1] != 3
            ):
                raise ValueError("canonical_anchors must have shape [A,3]")
            if canonical_anchors.shape[0] != dressable_model.anchor_features.shape[0]:
                raise ValueError("canonical_anchors must match the anchor topology")
            if not torch.is_floating_point(canonical_anchors) or not torch.isfinite(
                canonical_anchors
            ).all():
                raise ValueError("canonical_anchors must be finite and floating point")
            anchors = canonical_anchors.detach().clone().to(device=device, dtype=dtype)
        self.register_buffer("canonical_anchors", anchors)
        self.canonical_clothing_completer: CanonicalClothingCompleter | None = None
        self.register_buffer("anchor_graph_indices", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("anchor_graph_weights", torch.empty(0, device=device, dtype=dtype), persistent=False)

    def initialize_online_completion(
        self, anchor_graph_indices: torch.Tensor, anchor_graph_weights: torch.Tensor,
        hidden_dim: int = 128, num_blocks: int = 4,
    ) -> None:
        A=self.canonical_anchors.shape[0]
        validate_anchor_graph(anchor_graph_indices,anchor_graph_weights,A,anchor_graph_indices.shape[1])
        local_dim=self.multiview_aggregator.output_dim
        global_dim=self.dressable_model.clothing_film_generator.clothing_dim
        self.canonical_clothing_completer=CanonicalClothingCompleter(
            local_dim,global_dim,3,hidden_dim,num_blocks
        ).to(device=self.canonical_anchors.device,dtype=self.canonical_anchors.dtype)
        self.anchor_graph_indices=anchor_graph_indices.to(self.canonical_anchors.device)
        self.anchor_graph_weights=anchor_graph_weights.to(self.canonical_anchors)

    def encode_clothing_online(
        self, reference_images: torch.Tensor, reference_cloth_masks: torch.Tensor,
        reference_foreground_masks: torch.Tensor, reference_poses: torch.Tensor,
        reference_cameras: Sequence[dict[str, Any]], reference_valid_mask: torch.Tensor,
        deformation_fn=None, surface_depth_maps=None, surface_alpha_maps=None,
        require_depth_visibility: bool = True,
    ) -> dict[str, Any]:
        """Create online reference-only evidence and learned canonical completion."""

        if self.canonical_clothing_completer is None or self.anchor_graph_indices.numel()==0:
            raise RuntimeError("initialize online completion before formal forward")
        if reference_valid_mask.float().sum()<=0:
            raise ValueError("formal online forward rejects all-invalid references")
        zero_clothing_mask=bool(reference_cloth_masks.count_nonzero().item()==0)
        if zero_clothing_mask:
            encoded=self.encode_global_clothing(
                reference_images,torch.ones_like(reference_cloth_masks),reference_valid_mask
            )
            encoded={**encoded,
                "global_clothing_embedding":torch.zeros_like(encoded["global_clothing_embedding"]),
                "feature_maps":torch.zeros_like(encoded["feature_maps"]),
                "feature_cloth_masks":torch.zeros_like(encoded["feature_cloth_masks"]),
            }
        else:
            encoded=self.encode_global_clothing(reference_images,reference_cloth_masks,reference_valid_mask)
        projection=self.anchor_image_projector.project_and_sample(
            canonical_anchors=self.canonical_anchors,feature_maps=encoded["feature_maps"],
            reference_poses=reference_poses,reference_cameras=reference_cameras,
            image_height=reference_images.shape[-2],image_width=reference_images.shape[-1],
            feature_cloth_masks=encoded["feature_cloth_masks"],
            reference_cloth_masks=reference_cloth_masks,
            reference_foreground_masks=reference_foreground_masks,
            deformation_fn=deformation_fn,surface_depth_maps=surface_depth_maps,
            surface_alpha_maps=surface_alpha_maps,require_depth_visibility=require_depth_visibility,
        )
        observed=self.multiview_aggregator.aggregate_online_observations(
            projection["sampled_features"],projection["per_view_visibility"],
            projection["per_view_cloth_probability"],reference_valid_mask,
        )
        anchors=self.canonical_anchors
        center=anchors.mean(0,keepdim=True); scale=(anchors-center).abs().amax().clamp_min(1e-8)
        completion=self.canonical_clothing_completer(
            encoded["global_clothing_embedding"],observed["observed_surface_feature"],
            observed["observed_clothing_feature"],observed["observed_clothing_probability"],
            observed["observation_coverage"],(anchors-center)/scale,
            self.anchor_graph_indices,self.anchor_graph_weights,
        )
        return {"global_clothing_embedding":encoded["global_clothing_embedding"],
                "completion":completion,"projection":projection,"observed":observed}

    def compute_online_six_channel_residuals(
        self,
        *,
        protected_gaussian_mask: torch.Tensor | None = None,
        **online_inputs,
    ) -> dict[str, Any]:
        """Formal Module-3 path; no teacher, target image, cloth ID, or temporary gate."""

        forbidden={"teacher","target_rgb","target_foreground_mask","target_clothing_mask","cloth_id","reference_only_gate"}.intersection(online_inputs)
        if forbidden: raise ValueError(f"forbidden online conditioning fields: {sorted(forbidden)}")
        online=self.encode_clothing_online(**online_inputs); completion=online["completion"]
        raw,bounded,gated=self.dressable_model.compute_film_anchor_residuals(
            completion.gate_bundle(),clothing_embedding=online["global_clothing_embedding"],
            anchor_clothing_features=completion.completed_anchor_features,
        )
        gaussian=self.dressable_model.interpolate_anchor_clothing_residuals(gated)
        if protected_gaussian_mask is not None:
            gaussian=apply_protected_full_residual_guard(gaussian, protected_gaussian_mask)
        return {**online,"raw_anchor_residuals":raw,"bounded_anchor_residuals":bounded,
                "gated_anchor_residuals":gated,"gaussian_residuals":gaussian,
                "protected_full_residual_guard_applied":protected_gaussian_mask is not None}

    def encode_global_clothing(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a reference set without anchor projection."""

        return self.clothing_observation_encoder(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
        )

    def encode_clothing(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Backward-compatible alias for global-only clothing encoding."""

        return self.encode_global_clothing(
            reference_images, reference_cloth_masks, reference_valid_mask
        )

    def encode_clothing_with_anchors(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_poses: torch.Tensor,
        reference_cameras: Sequence[dict[str, Any]],
        canonical_anchors: torch.Tensor,
        canonical_normals: torch.Tensor | None = None,
        reference_valid_mask: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
    ) -> dict[str, Any]:
        """Encode global clothing and aggregate projected per-anchor features."""

        if canonical_anchors.shape[0] != self.dressable_model.anchor_features.shape[0]:
            raise ValueError(
                "canonical anchor count must match DressableGaussianModel anchor topology"
            )
        encoded = self.encode_global_clothing(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
        )
        projection = self.anchor_image_projector.project_and_sample(
            canonical_anchors=canonical_anchors,
            feature_maps=encoded["feature_maps"],
            reference_poses=reference_poses,
            reference_cameras=reference_cameras,
            image_height=reference_images.shape[-2],
            image_width=reference_images.shape[-1],
            feature_cloth_masks=encoded["feature_cloth_masks"],
            canonical_normals=canonical_normals,
            deformation_fn=deformation_fn,
            surface_depth_maps=surface_depth_maps,
            surface_alpha_maps=surface_alpha_maps,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_alpha_threshold=depth_alpha_threshold,
            require_depth_visibility=require_depth_visibility,
        )
        aggregated = self.multiview_aggregator(
            projection["sampled_features"],
            projection["visibility_confidence"],
            global_view_weights=encoded["view_weights"],
        )
        return {
            "global_clothing_embedding": encoded["global_clothing_embedding"],
            "anchor_clothing_features": aggregated["anchor_clothing_features"],
            "anchor_visibility": aggregated["anchor_visibility"],
            "projection": projection,
        }

    def compute_anchor_offsets_from_images(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
        reference_poses: torch.Tensor | None = None,
        reference_cameras: Sequence[dict[str, Any]] | None = None,
        canonical_anchors: torch.Tensor | None = None,
        canonical_normals: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict offsets through either explicit global-only or global+local path."""

        global_embedding, local_features = self._resolve_image_condition(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
            reference_poses,
            reference_cameras,
            canonical_anchors,
            canonical_normals,
            deformation_fn,
            surface_depth_maps,
            surface_alpha_maps,
            depth_abs_tolerance,
            depth_rel_tolerance,
            depth_alpha_threshold,
            require_depth_visibility,
        )
        return self.dressable_model.compute_film_anchor_offsets(
            clothing_embedding=global_embedding,
            anchor_clothing_features=local_features,
        )

    def get_dressed_canonical_params_from_images(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
        reference_poses: torch.Tensor | None = None,
        reference_cameras: Sequence[dict[str, Any]] | None = None,
        canonical_anchors: torch.Tensor | None = None,
        canonical_normals: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compose image-conditioned raw canonical Gaussian parameters."""

        global_embedding, local_features = self._resolve_image_condition(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
            reference_poses,
            reference_cameras,
            canonical_anchors,
            canonical_normals,
            deformation_fn,
            surface_depth_maps,
            surface_alpha_maps,
            depth_abs_tolerance,
            depth_rel_tolerance,
            depth_alpha_threshold,
            require_depth_visibility,
        )
        return self.dressable_model.get_dressed_canonical_params(
            clothing_embedding=global_embedding,
            anchor_clothing_features=local_features,
        )

    def render_from_images(
        self,
        viewpoint_camera,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
        reference_poses: torch.Tensor | None = None,
        reference_cameras: Sequence[dict[str, Any]] | None = None,
        canonical_anchors: torch.Tensor | None = None,
        canonical_normals: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
        **render_kwargs: Any,
    ):
        """Render through the original base with global or global+local conditioning."""

        global_embedding, local_features = self._resolve_image_condition(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
            reference_poses,
            reference_cameras,
            canonical_anchors,
            canonical_normals,
            deformation_fn,
            surface_depth_maps,
            surface_alpha_maps,
            depth_abs_tolerance,
            depth_rel_tolerance,
            depth_alpha_threshold,
            require_depth_visibility,
        )
        return self.dressable_model.render(
            viewpoint_camera,
            clothing_embedding=global_embedding,
            anchor_clothing_features=local_features,
            **render_kwargs,
        )

    def forward(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
        reference_poses: torch.Tensor | None = None,
        reference_cameras: Sequence[dict[str, Any]] | None = None,
        canonical_anchors: torch.Tensor | None = None,
        canonical_normals: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return dressed raw canonical parameters for the selected image path."""

        return self.get_dressed_canonical_params_from_images(
            reference_images,
            reference_cloth_masks,
            reference_valid_mask,
            reference_poses,
            reference_cameras,
            canonical_anchors,
            canonical_normals,
            deformation_fn,
            surface_depth_maps,
            surface_alpha_maps,
            depth_abs_tolerance,
            depth_rel_tolerance,
            depth_alpha_threshold,
            require_depth_visibility,
        )

    def forward_episode(
        self,
        episode: dict[str, Any],
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        render_target: bool = False,
        render_kwargs: dict[str, Any] | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.02,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
        require_mmlphuman_state_transaction: bool = False,
        reference_geometry_diagnostics: dict[str, Any] | None = None,
        reference_only_gate: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run one reference-target episode without using its cloth identifier."""

        if not isinstance(episode, dict):
            raise TypeError("episode must be a dictionary")
        required = (
            "reference_images",
            "reference_cloth_masks",
            "reference_poses",
            "reference_cameras",
            "reference_valid_mask",
        )
        missing = [key for key in required if key not in episode]
        if missing:
            raise KeyError(f"episode is missing required fields: {missing}")
        anchor_target = episode.get("anchor_offset_target")
        canonical_anchors = episode.get("canonical_anchors")
        if canonical_anchors is None and isinstance(anchor_target, dict):
            canonical_anchors = anchor_target.get("anchor_xyz")
        if canonical_anchors is None and self.canonical_anchors.numel() > 0:
            canonical_anchors = self.canonical_anchors
        if canonical_anchors is None:
            raise ValueError(
                "forward_episode requires canonical_anchors or anchor_offset_target.anchor_xyz"
            )
        device = self.dressable_model.anchor_features.device
        dtype = self.dressable_model.anchor_features.dtype
        reference_images = self._to_model_float(
            episode["reference_images"], device, dtype, "reference_images"
        )
        reference_masks = self._to_model_float(
            episode["reference_cloth_masks"], device, dtype, "reference_cloth_masks"
        )
        reference_poses = self._to_model_float(
            episode["reference_poses"], device, dtype, "reference_poses"
        )
        reference_valid = self._to_model_float(
            episode["reference_valid_mask"], device, dtype, "reference_valid_mask"
        )
        anchors = self._to_model_float(
            canonical_anchors, device, dtype, "canonical_anchors"
        )
        encoded = self.encode_clothing_with_anchors(
            reference_images=reference_images,
            reference_cloth_masks=reference_masks,
            reference_poses=reference_poses,
            reference_cameras=episode["reference_cameras"],
            canonical_anchors=anchors,
            canonical_normals=(
                None
                if episode.get("canonical_normals") is None
                else self._to_model_float(
                    episode["canonical_normals"], device, dtype, "canonical_normals"
                )
            ),
            reference_valid_mask=reference_valid,
            deformation_fn=deformation_fn,
            surface_depth_maps=surface_depth_maps,
            surface_alpha_maps=surface_alpha_maps,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_alpha_threshold=depth_alpha_threshold,
            require_depth_visibility=require_depth_visibility,
        )
        anchor_offsets = self.dressable_model.compute_film_anchor_offsets(
            clothing_embedding=encoded["global_clothing_embedding"],
            anchor_clothing_features=encoded["anchor_clothing_features"],
        )
        if reference_only_gate is not None:
            gate = self._to_model_float(
                reference_only_gate, device, dtype, "reference_only_gate"
            ).reshape(-1, 1)
            if gate.shape != (anchors.shape[0], 1):
                raise ValueError(
                    f"reference_only_gate has shape {tuple(gate.shape)}, "
                    f"expected {(anchors.shape[0], 1)}"
                )
            if ((gate < 0) | (gate > 1)).any():
                raise ValueError("reference_only_gate must be in [0,1]")
            anchor_offsets = {
                key: value * gate for key, value in anchor_offsets.items()
            }
        gaussian_offsets = self.dressable_model.interpolate_anchor_offsets_to_gaussians(
            anchor_offsets
        )
        base_model = self.dressable_model.base_model
        dressed = {
            "xyz": base_model._xyz + gaussian_offsets["delta_xyz"],
            "scaling": base_model._scaling + gaussian_offsets["delta_scaling"],
            "opacity": base_model._opacity + gaussian_offsets["delta_opacity"],
            "rotation": base_model._rotation,
            "sh0": base_model._sh0,
            "shN": base_model._shN,
        }
        rendered = None
        target_state_cache_restored = None
        if render_target:
            target_required = (
                "target_rgb",
                "target_pose",
                "target_Rh",
                "target_Th",
                "target_camera",
            )
            missing_target = [key for key in target_required if key not in episode]
            if missing_target:
                raise KeyError(f"render target is missing fields: {missing_target}")
            target_pose = self._to_model_float(
                episode["target_pose"], device, dtype, "target_pose"
            )
            target_Rh = self._to_model_float(
                episode["target_Rh"], device, dtype, "target_Rh"
            )
            target_Th = self._to_model_float(
                episode["target_Th"], device, dtype, "target_Th"
            )
            height, width = episode["target_rgb"].shape[-2:]
            camera = build_mmlphuman_camera(
                episode["target_camera"], height, width, device
            )
            base_model = self.dressable_model.base_model
            complete_state = all(
                hasattr(base_model, name)
                for name in ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th", "cache_dict")
            )
            if require_mmlphuman_state_transaction and not complete_state:
                raise RuntimeError(
                    "real target rendering requires complete MMLPHuman pose/Rh/Th/cache state"
                )
            if complete_state:
                original_cache = base_model.cache_dict
                original_state = {
                    name: getattr(base_model, name)
                    for name in ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
                }
                with mmlphuman_state_transaction(
                    base_model,
                    target_pose,
                    target_Rh,
                    target_Th,
                ):
                    rendered = base_model.render(
                        camera,
                        canonical_overrides=dressed,
                        **({} if render_kwargs is None else dict(render_kwargs)),
                    )
                target_state_cache_restored = base_model.cache_dict is original_cache and all(
                    getattr(base_model, name) is value
                    for name, value in original_state.items()
                )
                if not target_state_cache_restored:
                    raise RuntimeError("target render polluted MMLPHuman pose/cache state")
            else:
                previous_pose = getattr(base_model, "smpl_poses", None)
                previous_pose = (
                    previous_pose.clone()
                    if isinstance(previous_pose, torch.Tensor)
                    else previous_pose
                )
                try:
                    if hasattr(base_model, "smpl_poses"):
                        base_model.smpl_poses = target_pose.detach().cpu()
                    rendered = base_model.render(
                        camera,
                        canonical_overrides=dressed,
                        **({} if render_kwargs is None else dict(render_kwargs)),
                    )
                finally:
                    if isinstance(previous_pose, torch.Tensor):
                        base_model.smpl_poses = previous_pose
            if not isinstance(rendered, tuple) or not rendered:
                raise TypeError("base render must return a non-empty tuple")
            render_tensors = [value for value in rendered if isinstance(value, torch.Tensor)]
            if not render_tensors or any(not torch.isfinite(value).all() for value in render_tensors):
                raise FloatingPointError("target render contains NaN or Inf")
        return {
            "global_clothing_embedding": encoded["global_clothing_embedding"],
            "anchor_clothing_features": encoded["anchor_clothing_features"],
            "anchor_visibility": encoded["anchor_visibility"],
            "projection": encoded["projection"],
            "reference_geometry_diagnostics": reference_geometry_diagnostics,
            "anchor_offsets": anchor_offsets,
            "gaussian_offsets": gaussian_offsets,
            "dressed_canonical_params": dressed,
            "render": rendered,
            "target_state_cache_restored": target_state_cache_restored,
        }

    @staticmethod
    def _to_model_float(
        value: Any,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return value.to(device=device, dtype=dtype)

    def _resolve_image_condition(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None,
        reference_poses: torch.Tensor | None,
        reference_cameras: Sequence[dict[str, Any]] | None,
        canonical_anchors: torch.Tensor | None,
        canonical_normals: torch.Tensor | None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None,
        surface_depth_maps: torch.Tensor | None,
        surface_alpha_maps: torch.Tensor | None,
        depth_abs_tolerance: float,
        depth_rel_tolerance: float,
        depth_alpha_threshold: float,
        require_depth_visibility: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_required = (
            reference_poses is not None,
            reference_cameras is not None,
            canonical_anchors is not None,
        )
        if any(local_required) and not all(local_required):
            raise ValueError(
                "reference_poses, reference_cameras, and canonical_anchors must be "
                "provided together for local conditioning"
            )
        if not any(local_required):
            if (
                canonical_normals is not None
                or deformation_fn is not None
                or surface_depth_maps is not None
                or surface_alpha_maps is not None
                or require_depth_visibility
            ):
                raise ValueError(
                    "normals/deformation/depth visibility require the explicit local path"
                )
            encoded = self.encode_global_clothing(
                reference_images,
                reference_cloth_masks,
                reference_valid_mask,
            )
            return encoded["global_clothing_embedding"], None
        encoded = self.encode_clothing_with_anchors(
            reference_images=reference_images,
            reference_cloth_masks=reference_cloth_masks,
            reference_poses=reference_poses,
            reference_cameras=reference_cameras,
            canonical_anchors=canonical_anchors,
            canonical_normals=canonical_normals,
            reference_valid_mask=reference_valid_mask,
            deformation_fn=deformation_fn,
            surface_depth_maps=surface_depth_maps,
            surface_alpha_maps=surface_alpha_maps,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_alpha_threshold=depth_alpha_threshold,
            require_depth_visibility=require_depth_visibility,
        )
        return (
            encoded["global_clothing_embedding"],
            encoded["anchor_clothing_features"],
        )
