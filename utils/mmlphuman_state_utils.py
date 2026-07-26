from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


def _validate_state_tensor(
    value: torch.Tensor,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


@contextmanager
def mmlphuman_state_transaction(
    base_model: Any,
    pose: torch.Tensor,
    Rh: torch.Tensor,
    Th: torch.Tensor,
) -> Iterator[None]:
    """Temporarily install one complete MMLPHuman pose/global-transform state.

    The original ``cache_dict`` object and its entries are restored even when
    rendering raises. Internal pose tensors are restored by reference so this
    helper does not trigger setters or build a second cache during teardown.
    """

    pose = _validate_state_tensor(pose, "pose", (165,))
    Rh = _validate_state_tensor(Rh, "Rh", (3, 3))
    Th = _validate_state_tensor(Th, "Th", (3,))

    state_names = ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
    missing = [name for name in state_names if not hasattr(base_model, name)]
    if missing:
        raise AttributeError(f"base_model is missing restorable state: {missing}")
    original_cache = getattr(base_model, "cache_dict", None)
    if not isinstance(original_cache, dict):
        raise TypeError("base_model.cache_dict must be a dictionary")
    original_cache_contents = dict(original_cache)
    original_state = {name: getattr(base_model, name) for name in state_names}

    try:
        base_model.smpl_poses = pose.detach()
        base_model.Rh = Rh.detach()
        base_model.Th = Th.detach()
        active_cache = getattr(base_model, "cache_dict", None)
        if not isinstance(active_cache, dict):
            raise TypeError("MMLPHuman setters must leave cache_dict as a dictionary")
        active_cache.clear()
        yield
    finally:
        for name, value in original_state.items():
            setattr(base_model, name, value)
        original_cache.clear()
        original_cache.update(original_cache_contents)
        base_model.cache_dict = original_cache
