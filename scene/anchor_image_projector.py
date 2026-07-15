from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from utils.mmlphuman_anchor_visibility import compute_anchor_depth_visibility


class AnchorImageProjector(nn.Module):
    """Project canonical anchors and sample local image features.

    Cameras use a world-to-camera convention: ``x_cam = R @ x_world + T``.
    When normals are supplied they are treated as already posed, transformed
    by camera rotation, and scored with the front-facing cosine
    ``clamp(dot(n_cam, direction_to_camera), 0, 1)``.

    When official gsplat expected-depth maps are provided, visibility also
    rejects anchors whose camera-z differs from the sampled surface depth.
    """

    def __init__(self, align_corners: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        if not isinstance(align_corners, bool):
            raise TypeError("align_corners must be a bool")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
            raise ValueError("eps must be a positive number")
        self.align_corners = align_corners
        self.eps = float(eps)

    def project_and_sample(
        self,
        canonical_anchors: torch.Tensor,
        feature_maps: torch.Tensor,
        reference_poses: torch.Tensor,
        reference_cameras: Sequence[dict[str, Any]],
        image_height: int,
        image_width: int,
        feature_cloth_masks: torch.Tensor | None = None,
        reference_cloth_masks: torch.Tensor | None = None,
        reference_foreground_masks: torch.Tensor | None = None,
        canonical_normals: torch.Tensor | None = None,
        deformation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        surface_depth_maps: torch.Tensor | None = None,
        surface_alpha_maps: torch.Tensor | None = None,
        depth_abs_tolerance: float = 0.01,
        depth_rel_tolerance: float = 0.01,
        depth_alpha_threshold: float = 1e-4,
        require_depth_visibility: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Deform, project, and sample features with optional depth visibility."""

        num_views, num_anchors = self._validate_inputs(
            canonical_anchors,
            feature_maps,
            reference_poses,
            reference_cameras,
            image_height,
            image_width,
            feature_cloth_masks,
            canonical_normals,
            deformation_fn,
        )
        device, dtype = feature_maps.device, feature_maps.dtype
        if not isinstance(require_depth_visibility, bool):
            raise TypeError("require_depth_visibility must be a bool")
        if surface_depth_maps is None and surface_alpha_maps is not None:
            raise ValueError("surface_alpha_maps requires surface_depth_maps")
        if surface_depth_maps is None and require_depth_visibility:
            raise RuntimeError(
                "surface_depth_maps is required when depth visibility is enabled"
            )
        anchors = canonical_anchors.to(device=device, dtype=dtype)
        poses = reference_poses.to(device=device, dtype=dtype)
        normals = (
            None
            if canonical_normals is None
            else canonical_normals.to(device=device, dtype=dtype)
        )

        posed_views = []
        intrinsics = []
        rotations = []
        translations = []
        for view_index in range(num_views):
            posed = (
                anchors
                if deformation_fn is None
                else deformation_fn(anchors, poses[view_index], view_index)
            )
            self._validate_deformed_anchors(posed, anchors, view_index)
            intrinsic, rotation, translation = self._parse_camera(
                reference_cameras[view_index], device, dtype, view_index
            )
            posed_views.append(posed)
            intrinsics.append(intrinsic)
            rotations.append(rotation)
            translations.append(translation)

        posed_anchors = torch.stack(posed_views, dim=0)
        intrinsic_tensor = torch.stack(intrinsics, dim=0)
        rotation_tensor = torch.stack(rotations, dim=0)
        translation_tensor = torch.stack(translations, dim=0)
        camera_points = torch.einsum(
            "kij,kaj->kai", rotation_tensor, posed_anchors
        ) + translation_tensor[:, None, :]
        depth = camera_points[..., 2:3]
        positive_depth = depth > self.eps
        safe_depth = torch.where(
            depth.abs() > self.eps, depth, torch.ones_like(depth)
        )
        normalized_xy = camera_points[..., :2] / safe_depth
        pixel_x = (
            intrinsic_tensor[:, None, 0, 0] * normalized_xy[..., 0]
            + intrinsic_tensor[:, None, 0, 2]
        )
        pixel_y = (
            intrinsic_tensor[:, None, 1, 1] * normalized_xy[..., 1]
            + intrinsic_tensor[:, None, 1, 2]
        )
        pixels = torch.stack((pixel_x, pixel_y), dim=-1)
        in_frame = (
            (pixels[..., 0:1] >= 0)
            & (pixels[..., 0:1] <= image_width - 1)
            & (pixels[..., 1:2] >= 0)
            & (pixels[..., 1:2] <= image_height - 1)
        )
        sampling_grid = self._pixels_to_grid(
            pixels, image_height=image_height, image_width=image_width
        )
        grid_for_sample = sampling_grid.unsqueeze(2)
        sampled = F.grid_sample(
            feature_maps,
            grid_for_sample,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        sampled_features = sampled.squeeze(-1).permute(0, 2, 1)
        geometric_valid = in_frame & positive_depth
        sampled_features = sampled_features * geometric_valid.to(dtype=dtype)

        if feature_cloth_masks is None:
            cloth_confidence = torch.ones(
                num_views, num_anchors, 1, device=device, dtype=dtype
            )
        else:
            masks = feature_cloth_masks.to(device=device, dtype=dtype).clamp(0, 1)
            cloth_confidence = F.grid_sample(
                masks,
                grid_for_sample,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            ).squeeze(-1).permute(0, 2, 1).clamp(0, 1)

        def sample_image_mask(value: torch.Tensor | None, name: str) -> torch.Tensor:
            if value is None:
                return torch.ones(num_views,num_anchors,1,device=device,dtype=dtype)
            mask=value.to(device=device,dtype=dtype)
            if tuple(mask.shape)!=(num_views,1,image_height,image_width):
                raise ValueError(f"{name} must have shape [K,1,H,W]")
            if not torch.isfinite(mask).all() or ((mask<0)|(mask>1)).any():
                raise ValueError(f"{name} must be finite in [0,1]")
            return F.grid_sample(mask,grid_for_sample,mode="bilinear",padding_mode="zeros",align_corners=self.align_corners).squeeze(-1).permute(0,2,1).clamp(0,1)

        online_cloth_probability=sample_image_mask(reference_cloth_masks,"reference_cloth_masks")
        foreground_confidence=sample_image_mask(reference_foreground_masks,"reference_foreground_masks")

        if normals is None:
            angle_confidence = torch.ones(
                num_views, num_anchors, 1, device=device, dtype=dtype
            )
        else:
            normal_camera = torch.einsum("kij,aj->kai", rotation_tensor, normals)
            normal_camera = F.normalize(normal_camera, dim=-1, eps=self.eps)
            direction_to_camera = F.normalize(-camera_points, dim=-1, eps=self.eps)
            angle_confidence = (
                normal_camera * direction_to_camera
            ).sum(dim=-1, keepdim=True).clamp(0, 1)

        in_frame_float = in_frame.to(dtype=dtype)
        positive_depth_float = positive_depth.to(dtype=dtype)
        visibility = (
            in_frame_float
            * positive_depth_float
            * cloth_confidence
            * angle_confidence
        )
        foreground_mask_hit = geometric_valid & (cloth_confidence >= 0.5)
        depth_outputs: dict[str, torch.Tensor] = {}
        if surface_depth_maps is not None:
            depth_maps = surface_depth_maps.to(device=device, dtype=dtype)
            alpha_maps = (
                None
                if surface_alpha_maps is None
                else surface_alpha_maps.to(device=device, dtype=dtype)
            )
            depth_outputs = compute_anchor_depth_visibility(
                anchor_depth=depth,
                surface_depth_maps=depth_maps,
                sampling_grid=sampling_grid,
                positive_depth_mask=positive_depth,
                in_frame_mask=in_frame,
                foreground_mask_hit=foreground_mask_hit,
                abs_tolerance=depth_abs_tolerance,
                rel_tolerance=depth_rel_tolerance,
                align_corners=self.align_corners,
                surface_alpha_maps=alpha_maps,
                min_surface_alpha=depth_alpha_threshold,
            )
            visibility = visibility * depth_outputs["depth_confidence"]
        online_visibility = in_frame_float*positive_depth_float*angle_confidence*foreground_confidence
        if "depth_confidence" in depth_outputs:
            online_visibility=online_visibility*depth_outputs["depth_confidence"]
        outputs = {
            "sampled_features": sampled_features,
            "projected_pixels": pixels,
            "sampling_grid": sampling_grid,
            "projected_depth": depth,
            "in_frame_mask": in_frame_float,
            "positive_depth_mask": positive_depth_float,
            "cloth_mask_confidence": cloth_confidence,
            "foreground_mask_hit": foreground_mask_hit.to(dtype=dtype),
            "view_angle_confidence": angle_confidence,
            "visibility_confidence": visibility,
            "per_view_cloth_probability": online_cloth_probability,
            "per_view_visibility": online_visibility.clamp(0,1),
            "foreground_confidence": foreground_confidence,
        }
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise FloatingPointError("anchor projection produced NaN or Inf")
        outputs.update(depth_outputs)
        return outputs

    def _pixels_to_grid(
        self,
        pixels: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        if self.align_corners:
            if image_width <= 1 or image_height <= 1:
                raise ValueError("align_corners=True requires image dimensions greater than one")
            grid_x = 2 * pixels[..., 0] / (image_width - 1) - 1
            grid_y = 2 * pixels[..., 1] / (image_height - 1) - 1
        else:
            grid_x = 2 * (pixels[..., 0] + 0.5) / image_width - 1
            grid_y = 2 * (pixels[..., 1] + 0.5) / image_height - 1
        return torch.stack((grid_x, grid_y), dim=-1)

    @staticmethod
    def _parse_camera(
        camera: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        view_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(camera, dict):
            raise TypeError(f"reference_cameras[{view_index}] must be a dict")
        if "K" in camera:
            intrinsic_value = camera["K"]
        elif "intrinsics" in camera:
            intrinsic_value = camera["intrinsics"]
        else:
            raise KeyError(f"reference_cameras[{view_index}] requires K or intrinsics")

        if "R" in camera and "T" in camera:
            rotation = torch.as_tensor(camera["R"], device=device, dtype=dtype)
            translation = torch.as_tensor(camera["T"], device=device, dtype=dtype)
        elif "world_to_camera" in camera or "w2c" in camera:
            matrix_value = camera.get("world_to_camera", camera.get("w2c"))
            matrix = torch.as_tensor(matrix_value, device=device, dtype=dtype)
            if matrix.shape != (4, 4):
                raise ValueError(
                    f"reference_cameras[{view_index}] world_to_camera must be [4,4]"
                )
            if not torch.allclose(
                matrix[3], matrix.new_tensor([0, 0, 0, 1]), atol=1e-6, rtol=0
            ):
                raise ValueError("world_to_camera last row must be [0,0,0,1]")
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
        else:
            raise KeyError(
                f"reference_cameras[{view_index}] requires R/T or world_to_camera"
            )
        intrinsic = torch.as_tensor(intrinsic_value, device=device, dtype=dtype)
        if intrinsic.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                f"reference_cameras[{view_index}] has invalid intrinsic/extrinsic shapes"
            )
        if not all(
            torch.isfinite(value).all() for value in (intrinsic, rotation, translation)
        ):
            raise ValueError(f"reference_cameras[{view_index}] contains NaN or Inf")
        if intrinsic[0, 0].item() == 0 or intrinsic[1, 1].item() == 0:
            raise ValueError("camera focal lengths must be nonzero")
        return intrinsic, rotation, translation

    @staticmethod
    def _validate_deformed_anchors(
        posed: torch.Tensor,
        canonical: torch.Tensor,
        view_index: int,
    ) -> None:
        if not isinstance(posed, torch.Tensor):
            raise TypeError(f"deformation_fn view {view_index} must return a Tensor")
        if posed.shape != canonical.shape:
            raise ValueError(
                f"deformation_fn view {view_index} returned {tuple(posed.shape)}, "
                f"expected {tuple(canonical.shape)}"
            )
        if posed.device != canonical.device or posed.dtype != canonical.dtype:
            raise ValueError("deformation_fn output must match anchor device and dtype")
        if not torch.isfinite(posed).all():
            raise ValueError("deformation_fn output contains NaN or Inf")

    @staticmethod
    def _validate_inputs(
        canonical_anchors: torch.Tensor,
        feature_maps: torch.Tensor,
        reference_poses: torch.Tensor,
        reference_cameras: Sequence[dict[str, Any]],
        image_height: int,
        image_width: int,
        feature_cloth_masks: torch.Tensor | None,
        canonical_normals: torch.Tensor | None,
        deformation_fn: Any,
    ) -> tuple[int, int]:
        if not isinstance(feature_maps, torch.Tensor) or feature_maps.ndim != 4:
            raise ValueError("feature_maps must be a Tensor[K,C,Hf,Wf]")
        if not torch.is_floating_point(feature_maps) or not torch.isfinite(feature_maps).all():
            raise ValueError("feature_maps must be a finite floating-point tensor")
        num_views = feature_maps.shape[0]
        if num_views <= 0 or min(feature_maps.shape[1:]) <= 0:
            raise ValueError("feature_maps dimensions must be positive")
        if not isinstance(canonical_anchors, torch.Tensor):
            raise TypeError("canonical_anchors must be a torch.Tensor")
        if canonical_anchors.ndim != 2 or canonical_anchors.shape[1] != 3:
            raise ValueError("canonical_anchors must have shape [A,3]")
        if canonical_anchors.shape[0] <= 0 or not torch.is_floating_point(canonical_anchors):
            raise ValueError("canonical_anchors must be a non-empty floating-point tensor")
        if not torch.isfinite(canonical_anchors).all():
            raise ValueError("canonical_anchors contains NaN or Inf")
        if not isinstance(reference_poses, torch.Tensor) or reference_poses.ndim != 2:
            raise ValueError("reference_poses must be a Tensor[K,P]")
        if not torch.is_floating_point(reference_poses):
            raise TypeError("reference_poses must have a floating-point dtype")
        if reference_poses.shape[0] != num_views or not torch.isfinite(reference_poses).all():
            raise ValueError("reference_poses must have K finite rows")
        if not isinstance(reference_cameras, Sequence) or isinstance(
            reference_cameras, (str, bytes)
        ):
            raise TypeError("reference_cameras must be a sequence of camera dictionaries")
        if len(reference_cameras) != num_views:
            raise ValueError("reference_cameras length must match feature-map views")
        if not isinstance(image_height, int) or not isinstance(image_width, int):
            raise TypeError("image_height and image_width must be integers")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_height and image_width must be positive")
        if feature_cloth_masks is not None:
            expected = (num_views, 1, feature_maps.shape[2], feature_maps.shape[3])
            if not isinstance(feature_cloth_masks, torch.Tensor) or tuple(
                feature_cloth_masks.shape
            ) != expected:
                raise ValueError(f"feature_cloth_masks must have shape {expected}")
            if not torch.isfinite(feature_cloth_masks).all():
                raise ValueError("feature_cloth_masks contains NaN or Inf")
        if canonical_normals is not None:
            if not isinstance(canonical_normals, torch.Tensor) or canonical_normals.shape != canonical_anchors.shape:
                raise ValueError("canonical_normals must have shape [A,3]")
            if not torch.isfinite(canonical_normals).all():
                raise ValueError("canonical_normals contains NaN or Inf")
        if deformation_fn is not None and not callable(deformation_fn):
            raise TypeError("deformation_fn must be callable or None")
        return num_views, canonical_anchors.shape[0]
