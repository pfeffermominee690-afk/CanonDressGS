from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_anchor_lbs_grid(path: str | Path) -> dict[str, torch.Tensor]:
    """Load and validate an MMLPHuman ``lbs_weights_grid.npz`` archive."""

    grid_path = Path(path)
    if not grid_path.is_file():
        raise FileNotFoundError(f"anchor LBS grid does not exist: {grid_path}")
    with np.load(grid_path, allow_pickle=False) as archive:
        missing = sorted({"grid", "bbox_min", "bbox_max"}.difference(archive.files))
        if missing:
            raise KeyError(f"anchor LBS grid is missing keys: {missing}")
        grid_info = {
            name: torch.from_numpy(np.asarray(archive[name])).clone()
            for name in archive.files
        }
    _validate_lbs_grid_info(grid_info)
    return grid_info


def interpolate_anchor_lbs_weights(
    grid_info: Mapping[str, Any],
    canonical_anchors: torch.Tensor,
) -> torch.Tensor:
    """Interpolate anchor weights with MMLPHuman's exact grid-sampling convention.

    This is the device-agnostic equivalent of
    ``utils.smpl_utils.interpolate_skinningfield``: the volume is stored as
    ``[X,Y,Z,J]``, points are normalized by the stored bounding box, coordinates
    are reordered to ``[z,y,x]``, and 5-D ``grid_sample`` uses border padding
    with ``align_corners=True``.
    """

    _validate_anchor_tensor(canonical_anchors, "canonical_anchors")
    _validate_lbs_grid_info(grid_info)
    device, dtype = canonical_anchors.device, canonical_anchors.dtype
    grid = torch.as_tensor(grid_info["grid"], device=device, dtype=dtype)
    bbox_min = torch.as_tensor(grid_info["bbox_min"], device=device, dtype=dtype)
    bbox_max = torch.as_tensor(grid_info["bbox_max"], device=device, dtype=dtype)
    normalized = (canonical_anchors - bbox_min) / (bbox_max - bbox_min) * 2 - 1
    sample_points = normalized[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        grid.permute(3, 0, 1, 2).unsqueeze(0),
        sample_points,
        padding_mode="border",
        align_corners=True,
    )
    weights = sampled[0, :, 0, 0, :].transpose(0, 1).contiguous()
    if not torch.isfinite(weights).all():
        raise FloatingPointError("interpolated anchor LBS weights contain NaN or Inf")
    return weights


def validate_anchor_lbs_weights(
    weights: torch.Tensor,
    num_anchors: int,
    num_joints: int,
    sum_tolerance: float = 5e-4,
    negative_tolerance: float = 1e-7,
) -> None:
    """Validate fixed anchor skinning weights without silently renormalizing them."""

    if not isinstance(weights, torch.Tensor):
        raise TypeError("anchor_lbs_weights must be a torch.Tensor")
    if tuple(weights.shape) != (num_anchors, num_joints):
        raise ValueError(
            "anchor_lbs_weights must have shape "
            f"[{num_anchors},{num_joints}], got {tuple(weights.shape)}"
        )
    if not torch.is_floating_point(weights) or not torch.isfinite(weights).all():
        raise ValueError("anchor_lbs_weights must be finite and floating point")
    if torch.any(weights < -negative_tolerance):
        raise ValueError("anchor_lbs_weights contain negative values")
    row_sums = weights.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=sum_tolerance,
        rtol=0,
    ):
        raise ValueError(
            "anchor_lbs_weights row sums must be close to 1; "
            f"range=[{row_sums.min().item():.8g}, {row_sums.max().item():.8g}]"
        )


class MMLPHumanAnchorDeformationAdapter:
    """Deform checkpoint control anchors with the original MMLPHuman body path.

    Real mode is constructed only through :meth:`from_mmlphuman_base`. It uses
    checkpoint ``xyz_vt`` as the fixed control topology, evaluates the original
    pose-conditioned ``get_dxyz_vt`` and ``get_rigid_transform`` properties,
    and applies anchor weights interpolated from the original LBS volume.

    Explicit identity and external-callable modes are retained for synthetic
    tests. An unconfigured adapter never falls back to identity.
    """

    def __init__(
        self,
        base_model: Any,
        canonical_anchors: torch.Tensor,
        external_deformation_fn: Callable[
            [torch.Tensor, torch.Tensor, int | None], torch.Tensor
        ]
        | None = None,
        identity_mode: bool = False,
    ) -> None:
        _validate_anchor_tensor(canonical_anchors, "canonical_anchors")
        if external_deformation_fn is not None and not callable(external_deformation_fn):
            raise TypeError("external_deformation_fn must be callable or None")
        if identity_mode and external_deformation_fn is not None:
            raise ValueError("identity_mode and external_deformation_fn are mutually exclusive")
        self.base_model = base_model
        self.canonical_anchors = canonical_anchors.detach().clone()
        self.external_deformation_fn = external_deformation_fn
        self.identity_mode = bool(identity_mode)
        self.anchor_lbs_weights: torch.Tensor | None = None
        self.num_joints: int | None = None
        self.include_control_displacement = True
        self.reference_Rh: torch.Tensor | None = None
        self.reference_Th: torch.Tensor | None = None
        self._real_mode = False
        self._base_state_lock = RLock()

    @classmethod
    def identity(
        cls,
        canonical_anchors: torch.Tensor,
    ) -> "MMLPHumanAnchorDeformationAdapter":
        """Construct an explicit synthetic identity adapter."""

        return cls(None, canonical_anchors, identity_mode=True)

    @classmethod
    def from_mmlphuman_base(
        cls,
        base_model: Any,
        canonical_anchors: torch.Tensor | None = None,
        *,
        lbs_grid_info: Mapping[str, Any] | None = None,
        lbs_grid_path: str | Path | None = None,
        reference_Rh: torch.Tensor | None = None,
        reference_Th: torch.Tensor | None = None,
        include_control_displacement: bool = True,
        weight_sum_tolerance: float = 5e-4,
    ) -> "MMLPHumanAnchorDeformationAdapter":
        """Build the strict real adapter from a frozen restored GaussianModel."""

        if base_model is None:
            raise ValueError("base_model is required for real anchor deformation")
        if (lbs_grid_info is None) == (lbs_grid_path is None):
            raise ValueError("provide exactly one of lbs_grid_info or lbs_grid_path")
        base_anchors = getattr(base_model, "xyz_vt", None)
        _validate_anchor_tensor(base_anchors, "base_model.xyz_vt")
        anchors = base_anchors if canonical_anchors is None else canonical_anchors
        _validate_anchor_tensor(anchors, "canonical_anchors")
        if anchors.shape != base_anchors.shape:
            raise ValueError("canonical_anchors shape must match base_model.xyz_vt")
        if anchors.device != base_anchors.device or anchors.dtype != base_anchors.dtype:
            raise ValueError("canonical_anchors must match base_model.xyz_vt device and dtype")
        if not torch.allclose(anchors, base_anchors, atol=1e-6, rtol=0):
            raise ValueError(
                "real canonical_anchors must equal checkpoint xyz_vt in the same ordering"
            )
        t_joints = getattr(base_model, "t_joints", None)
        parents = getattr(base_model, "joint_parents", None)
        ac_inv = getattr(base_model, "Ac_inv", None)
        if not isinstance(t_joints, torch.Tensor) or t_joints.ndim != 2 or t_joints.shape[1] != 3:
            raise ValueError("base_model.t_joints must have shape [J,3]")
        num_joints = int(t_joints.shape[0])
        if not isinstance(parents, torch.Tensor) or parents.numel() != num_joints:
            raise ValueError("base_model.joint_parents must contain J entries")
        if not isinstance(ac_inv, torch.Tensor) or tuple(ac_inv.shape) != (num_joints, 4, 4):
            raise ValueError("base_model.Ac_inv must have shape [J,4,4]")
        gaussian_weights = getattr(base_model, "_weights", None)
        if not isinstance(gaussian_weights, torch.Tensor) or gaussian_weights.ndim != 2:
            raise ValueError("checkpoint must contain Gaussian _weights Tensor[N,J]")
        if gaussian_weights.shape[1] != num_joints:
            raise ValueError("checkpoint Gaussian joint count does not match t_joints")
        dxyz_vt = getattr(base_model, "dxyz_vt", None)
        if not isinstance(dxyz_vt, torch.Tensor) or dxyz_vt.shape != base_anchors.shape:
            raise ValueError("base_model.dxyz_vt must match xyz_vt shape")
        if getattr(type(base_model), "smpl_poses", None) is None:
            raise ValueError("base_model does not expose the MMLPHuman smpl_poses state")
        if (
            getattr(type(base_model), "get_dxyz_vt", None) is None
            or getattr(type(base_model), "get_rigid_transform", None) is None
        ):
            raise ValueError("base_model lacks required MMLPHuman deformation getters")
        _require_frozen_base(base_model)

        loaded_grid = load_anchor_lbs_grid(lbs_grid_path) if lbs_grid_path is not None else dict(lbs_grid_info)
        grid = torch.as_tensor(loaded_grid["grid"])
        if grid.ndim != 4 or grid.shape[-1] != num_joints:
            raise ValueError(
                "LBS grid joint dimension must match checkpoint; "
                f"grid={tuple(grid.shape)}, joints={num_joints}"
            )
        anchor_weights = interpolate_anchor_lbs_weights(loaded_grid, base_anchors)
        validate_anchor_lbs_weights(
            anchor_weights,
            base_anchors.shape[0],
            num_joints,
            sum_tolerance=weight_sum_tolerance,
        )

        adapter = cls(base_model, anchors)
        adapter.anchor_lbs_weights = anchor_weights.detach().clone()
        adapter.num_joints = num_joints
        adapter.include_control_displacement = bool(include_control_displacement)
        adapter._real_mode = True
        if reference_Rh is not None or reference_Th is not None:
            if reference_Rh is None or reference_Th is None:
                raise ValueError("reference_Rh and reference_Th must be provided together")
            adapter.set_reference_transforms(reference_Rh, reference_Th)
        return adapter

    def set_reference_transforms(
        self,
        reference_Rh: torch.Tensor,
        reference_Th: torch.Tensor,
    ) -> None:
        """Attach per-view global transforms used by projector-compatible calls."""

        if not isinstance(reference_Rh, torch.Tensor) or not isinstance(reference_Th, torch.Tensor):
            raise TypeError("reference_Rh and reference_Th must be tensors")
        rh = reference_Rh.unsqueeze(0) if reference_Rh.ndim == 2 else reference_Rh
        th = reference_Th.unsqueeze(0) if reference_Th.ndim == 1 else reference_Th
        if rh.ndim != 3 or rh.shape[1:] != (3, 3):
            raise ValueError("reference_Rh must have shape [K,3,3] or [3,3]")
        if th.ndim != 2 or th.shape[1] != 3 or th.shape[0] != rh.shape[0]:
            raise ValueError("reference_Th must have shape [K,3] matching reference_Rh")
        if not torch.is_floating_point(rh) or not torch.is_floating_point(th):
            raise TypeError("reference_Rh/reference_Th must be floating point")
        if not torch.isfinite(rh).all() or not torch.isfinite(th).all():
            raise ValueError("reference_Rh/reference_Th contain NaN or Inf")
        self.reference_Rh = rh.detach().clone()
        self.reference_Th = th.detach().clone()

    def deform(
        self,
        canonical_anchors: torch.Tensor,
        pose: torch.Tensor,
        view_index: int | None = None,
    ) -> torch.Tensor:
        """Return world-space posed anchors without persistent base mutation."""

        self._validate_call(canonical_anchors, pose)
        if self.identity_mode:
            output = canonical_anchors
        elif self.external_deformation_fn is not None:
            output = self.external_deformation_fn(canonical_anchors, pose, view_index)
        elif self._real_mode:
            if self.reference_Rh is None or self.reference_Th is None:
                raise RuntimeError(
                    "real projector deformation requires explicit reference_Rh/reference_Th"
                )
            if not isinstance(view_index, int):
                raise ValueError("real projector deformation requires an integer view_index")
            if view_index < 0 or view_index >= self.reference_Rh.shape[0]:
                raise IndexError("view_index is outside the stored reference transforms")
            output = self.deform_anchors(
                canonical_anchors,
                pose,
                self.reference_Rh[view_index],
                self.reference_Th[view_index],
            )
        else:
            raise RuntimeError(
                "real MMLPHuman anchor deformation is not configured; use "
                "from_mmlphuman_base(), provide external_deformation_fn, or use "
                "identity() only for synthetic tests"
            )
        self._validate_output(output, canonical_anchors)
        return output

    @torch.no_grad()
    def deform_anchors(
        self,
        canonical_anchors: torch.Tensor,
        pose: torch.Tensor,
        Rh: torch.Tensor,
        Th: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Execute the real checkpoint control-anchor deformation path.

        This method intentionally runs under ``no_grad``: the frozen backbone,
        fixed LBS weights, pose projection, and camera sampling geometry are not
        optimized by the current image-conditioned training stage.
        """

        if not self._real_mode or self.anchor_lbs_weights is None or self.num_joints is None:
            raise RuntimeError("deform_anchors requires from_mmlphuman_base()")
        self._validate_call(canonical_anchors, pose)
        if pose.numel() != self.num_joints * 3:
            raise ValueError(
                f"pose must contain {self.num_joints * 3} axis-angle values, "
                f"got {pose.numel()}"
            )
        rh = torch.as_tensor(Rh, device=canonical_anchors.device, dtype=canonical_anchors.dtype)
        th = torch.as_tensor(Th, device=canonical_anchors.device, dtype=canonical_anchors.dtype)
        _validate_global_transform(rh, th)
        displacement, rigid_transforms = self._evaluate_base_pose(pose)
        displacement = displacement.to(
            device=canonical_anchors.device, dtype=canonical_anchors.dtype
        )
        rigid_transforms = rigid_transforms.to(
            device=canonical_anchors.device, dtype=canonical_anchors.dtype
        )
        if displacement.shape != canonical_anchors.shape:
            raise ValueError("base get_dxyz_vt output does not match anchor topology")
        control_displacement = (
            displacement if self.include_control_displacement else torch.zeros_like(displacement)
        )
        displaced_canonical = canonical_anchors + control_displacement
        weights = self.anchor_lbs_weights.to(
            device=canonical_anchors.device, dtype=canonical_anchors.dtype
        )
        posed = self.apply_lbs_transforms(
            displaced_canonical,
            weights,
            rigid_transforms,
            rh,
            th,
        )
        self._validate_output(posed, canonical_anchors)
        if not return_diagnostics:
            return posed
        diagnostics = {
            "anchor_lbs_weights": weights,
            "lbs_weight_row_sums": weights.sum(dim=1),
            "control_displacement": control_displacement,
            "canonical_with_control_displacement": displaced_canonical,
        }
        return posed, diagnostics

    @staticmethod
    def apply_lbs_transforms(
        canonical_anchors: torch.Tensor,
        anchor_lbs_weights: torch.Tensor,
        rigid_transforms: torch.Tensor,
        Rh: torch.Tensor,
        Th: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``sum_j w_aj G_j`` followed by global ``Rh`` and ``Th``.

        Unlike :meth:`deform_anchors`, this pure tensor helper preserves
        autograd and is used by CPU synthetic tests of the mathematical core.
        """

        _validate_anchor_tensor(canonical_anchors, "canonical_anchors")
        if not isinstance(rigid_transforms, torch.Tensor) or rigid_transforms.ndim != 3:
            raise ValueError("rigid_transforms must have shape [J,4,4]")
        num_joints = rigid_transforms.shape[0]
        if tuple(rigid_transforms.shape[1:]) != (4, 4):
            raise ValueError("rigid_transforms must have shape [J,4,4]")
        validate_anchor_lbs_weights(
            anchor_lbs_weights,
            canonical_anchors.shape[0],
            num_joints,
        )
        for name, value in (
            ("anchor_lbs_weights", anchor_lbs_weights),
            ("rigid_transforms", rigid_transforms),
            ("Rh", Rh),
            ("Th", Th),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != canonical_anchors.device or value.dtype != canonical_anchors.dtype:
                raise ValueError(f"{name} must match anchor device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        _validate_global_transform(Rh, Th)
        expected_last_row = rigid_transforms.new_tensor([0, 0, 0, 1]).expand(num_joints, -1)
        if not torch.allclose(
            rigid_transforms[:, 3], expected_last_row, atol=1e-5, rtol=0
        ):
            raise ValueError("rigid_transforms must be affine homogeneous matrices")
        blended = torch.einsum("aj,jkl->akl", anchor_lbs_weights, rigid_transforms)
        homogeneous = F.pad(canonical_anchors, (0, 1), value=1)
        body_space = torch.einsum("aij,aj->ai", blended, homogeneous)[:, :3]
        world_space = torch.einsum("ij,aj->ai", Rh, body_space) + Th
        if not torch.isfinite(world_space).all():
            raise FloatingPointError("anchor deformation produced NaN or Inf")
        return world_space

    def _evaluate_base_pose(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate original getters and restore all transient base state."""

        base = self.base_model
        state_names = ("_smpl_poses", "smpl_poses_cuda", "cache_dict", "_Rh", "_Th")
        missing = [name for name in state_names if not hasattr(base, name)]
        if missing:
            raise ValueError(f"base model is missing restorable pose state: {missing}")
        with self._base_state_lock:
            previous = {name: getattr(base, name) for name in state_names}
            try:
                base.smpl_poses = pose.detach()
                displacement = base.get_dxyz_vt
                rigid_value = base.get_rigid_transform
                if not isinstance(rigid_value, (list, tuple)) or len(rigid_value) != 2:
                    raise ValueError("base get_rigid_transform must return [rotations, G]")
                rigid_transforms = rigid_value[1]
                if not isinstance(displacement, torch.Tensor):
                    raise TypeError("base get_dxyz_vt must return a tensor")
                if not isinstance(rigid_transforms, torch.Tensor):
                    raise TypeError("base rigid transforms must be a tensor")
                return displacement.detach().clone(), rigid_transforms.detach().clone()
            finally:
                for name, value in previous.items():
                    setattr(base, name, value)

    def _validate_call(
        self,
        canonical_anchors: torch.Tensor,
        pose: torch.Tensor,
    ) -> None:
        _validate_anchor_tensor(canonical_anchors, "canonical_anchors")
        if canonical_anchors.shape != self.canonical_anchors.shape:
            raise ValueError("canonical_anchors do not match adapter topology")
        reference = self.canonical_anchors.to(
            device=canonical_anchors.device, dtype=canonical_anchors.dtype
        )
        if not torch.allclose(canonical_anchors, reference, atol=1e-6, rtol=0):
            raise ValueError("canonical_anchors values do not match adapter initialization")
        if not isinstance(pose, torch.Tensor) or pose.ndim != 1:
            raise ValueError("pose must be a Tensor[P]")
        if not torch.is_floating_point(pose) or not torch.isfinite(pose).all():
            raise ValueError("pose must be finite and floating point")

    @staticmethod
    def _validate_output(output: torch.Tensor, canonical_anchors: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor) or output.shape != canonical_anchors.shape:
            raise ValueError("anchor deformation must return Tensor[A,3]")
        if output.device != canonical_anchors.device or output.dtype != canonical_anchors.dtype:
            raise ValueError("deformed anchors must match input device and dtype")
        if not torch.isfinite(output).all():
            raise ValueError("deformed anchors contain NaN or Inf")


def _validate_anchor_tensor(value: Any, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [A,3] with A > 0")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and floating point")


def _validate_lbs_grid_info(grid_info: Mapping[str, Any]) -> None:
    if not isinstance(grid_info, Mapping):
        raise TypeError("lbs_grid_info must be a mapping")
    missing = sorted({"grid", "bbox_min", "bbox_max"}.difference(grid_info))
    if missing:
        raise KeyError(f"lbs_grid_info is missing keys: {missing}")
    grid = torch.as_tensor(grid_info["grid"])
    bbox_min = torch.as_tensor(grid_info["bbox_min"])
    bbox_max = torch.as_tensor(grid_info["bbox_max"])
    if grid.ndim != 4 or min(grid.shape) <= 0:
        raise ValueError("LBS grid must have shape [X,Y,Z,J]")
    if not torch.is_floating_point(grid) or not torch.isfinite(grid).all():
        raise ValueError("LBS grid must be finite and floating point")
    if tuple(bbox_min.shape) != (3,) or tuple(bbox_max.shape) != (3,):
        raise ValueError("LBS bbox_min/bbox_max must have shape [3]")
    if not torch.isfinite(bbox_min).all() or not torch.isfinite(bbox_max).all():
        raise ValueError("LBS bounding box contains NaN or Inf")
    if torch.any(bbox_max <= bbox_min):
        raise ValueError("LBS bbox_max must be greater than bbox_min")
    if "grid_dims" in grid_info:
        grid_dims = torch.as_tensor(grid_info["grid_dims"]).reshape(-1)
        expected = torch.tensor(grid.shape[:3], dtype=grid_dims.dtype)
        if grid_dims.numel() != 3 or not torch.equal(grid_dims.cpu(), expected):
            raise ValueError("LBS grid_dims does not match grid spatial shape")


def _validate_global_transform(Rh: torch.Tensor, Th: torch.Tensor) -> None:
    if not isinstance(Rh, torch.Tensor) or tuple(Rh.shape) != (3, 3):
        raise ValueError("Rh must have shape [3,3]")
    if not isinstance(Th, torch.Tensor) or tuple(Th.shape) != (3,):
        raise ValueError("Th must have shape [3]")
    if not torch.is_floating_point(Rh) or not torch.is_floating_point(Th):
        raise TypeError("Rh and Th must be floating point")
    if not torch.isfinite(Rh).all() or not torch.isfinite(Th).all():
        raise ValueError("Rh or Th contains NaN or Inf")
    identity = torch.eye(3, device=Rh.device, dtype=Rh.dtype)
    if not torch.allclose(Rh.transpose(0, 1) @ Rh, identity, atol=1e-4, rtol=0):
        raise ValueError("Rh must be an orthonormal rotation matrix")
    if not torch.allclose(torch.linalg.det(Rh), Rh.new_tensor(1.0), atol=1e-4, rtol=0):
        raise ValueError("Rh must have determinant +1")


def _require_frozen_base(base_model: Any) -> None:
    from utils.dressable_checkpoint_utils import iter_base_parameters

    trainable = [parameter for parameter in iter_base_parameters(base_model) if parameter.requires_grad]
    if trainable:
        raise ValueError("base_model must be frozen before real anchor deformation")
