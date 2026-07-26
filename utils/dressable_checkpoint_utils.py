from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn


def resolve_mmlphuman_checkpoint(
    model_dir: str | Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Resolve an explicit checkpoint or the numerically largest chkpnt step."""

    model_path = Path(model_dir)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = model_path / path
        if not path.is_file():
            raise FileNotFoundError(f"MMLPHuman checkpoint does not exist: {path}")
        return path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MMLPHuman model directory does not exist: {model_path}")
    candidates = list(model_path.glob("chkpnt*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no chkpnt*.pth files found in {model_path}")

    def step_key(path: Path) -> tuple[int, str]:
        matches = re.findall(r"\d+", path.stem)
        return (int(matches[-1]) if matches else -1, path.name)

    return max(candidates, key=step_key).resolve()


def compute_base_fingerprint(checkpoint_path: str | Path, base_model: Any) -> str:
    """Hash checkpoint identity and core tensor shapes without modifying the file."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot fingerprint missing checkpoint: {path}")
    shape_parts = []
    for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN"):
        value = getattr(base_model, name, None)
        if not isinstance(value, torch.Tensor):
            raise AttributeError(f"base model is missing core tensor {name}")
        shape_parts.append(f"{name}:{tuple(value.shape)}")
    stat = path.stat()
    payload = "|".join((path.name, str(stat.st_size), *shape_parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_base_model(base_model: Any) -> None:
    """Freeze direct Parameters/tensors and nested nn.Modules on GaussianModel-like objects."""

    for parameter in iter_base_parameters(base_model):
        parameter.requires_grad_(False)
    for value in vars(base_model).values():
        if isinstance(value, nn.Module):
            value.eval()
    if isinstance(base_model, nn.Module):
        base_model.eval()


def iter_base_parameters(base_model: Any) -> Iterator[torch.Tensor]:
    """Yield unique trainable tensors from nn.Module and legacy GaussianModel state."""

    seen: set[int] = set()
    if isinstance(base_model, nn.Module):
        for parameter in base_model.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter
    for value in vars(base_model).values():
        candidates: list[Any]
        if isinstance(value, dict):
            candidates = list(value.values())
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            candidates = [value]
        for candidate in candidates:
            if isinstance(candidate, nn.Module):
                nested = candidate.parameters()
            elif isinstance(candidate, torch.Tensor) and candidate.requires_grad:
                nested = (candidate,)
            else:
                continue
            for parameter in nested:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter


def load_frozen_mmlphuman_base(
    model_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cuda",
):
    """Load a complete original checkpoint and attach immutable identity metadata.

    MMLPHuman's legacy ``GaussianModel`` is not an ``nn.Module`` and internally
    initializes CUDA-oriented state during ``restore``. This function preserves
    that implementation while freezing every discoverable parameter afterwards.
    """

    resolved = resolve_mmlphuman_checkpoint(model_dir, checkpoint_path)
    try:
        from scene.gaussian_model import GaussianModel
    except ImportError as error:
        raise RuntimeError(
            "loading a real MMLPHuman base requires the full gsplat/PyTorch3D environment"
        ) from error
    # Preserve the checkpoint's original mixed CPU/CUDA placement.
    # In particular, t_joints/all_poses are consumed by CPU/NumPy code.
    load_data = torch.load(resolved, weights_only=False)
    if not isinstance(load_data, dict) or "iteration" not in load_data:
        raise ValueError("MMLPHuman checkpoint must be a complete capture dict with iteration")
    from utils.smpl_utils import init_smpl_pose

    # Required before GaussianModel.restore():
    # restore() rebuilds body deformation state using smpl.smpl_bigpose.
    init_smpl_pose()

    model = GaussianModel()
    model.restore(load_data)
    required = (
        "_xyz",
        "_scaling",
        "_rotation",
        "_opacity",
        "_sh0",
        "_shN",
        "_weights",
        "xyz_vt",
        "xyz_ft",
        "nbr_gs",
        "nbr_gs_invdist",
        "dxyz_bs",
    )
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(f"checkpoint is missing required base/control state: {missing}")
    freeze_base_model(model)
    fingerprint = compute_base_fingerprint(resolved, model)
    model._dressable_checkpoint_path = str(resolved)
    model._dressable_checkpoint_step = int(load_data["iteration"])
    model._dressable_base_fingerprint = fingerprint

    # Match the original MMLPHuman test initialization sequence.
    model.is_test = True
    model.prepare_test()

    return model
