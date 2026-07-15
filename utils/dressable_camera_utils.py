from __future__ import annotations

from typing import Any

import torch


def validate_camera_data(camera_data: dict[str, Any]) -> None:
    """Validate MMLPHuman pinhole camera fields without guessing missing values."""

    if not isinstance(camera_data, dict):
        raise TypeError("camera_data must be a dictionary")
    if "K" not in camera_data and "intrinsics" not in camera_data:
        raise KeyError("camera_data requires K or intrinsics")
    has_w2c = "w2c" in camera_data
    has_rt = "R" in camera_data and "T" in camera_data
    if not has_w2c and not has_rt:
        raise KeyError("camera_data requires w2c or both R and T")
    intrinsic = torch.as_tensor(camera_data.get("K", camera_data.get("intrinsics")))
    if intrinsic.numel() != 9 or not torch.isfinite(intrinsic).all():
        raise ValueError("camera intrinsics must contain nine finite values")
    intrinsic = intrinsic.reshape(3, 3)
    if intrinsic[0, 0] == 0 or intrinsic[1, 1] == 0:
        raise ValueError("camera focal lengths must be nonzero")
    if has_w2c:
        w2c = torch.as_tensor(camera_data["w2c"])
        if w2c.numel() != 16 or not torch.isfinite(w2c).all():
            raise ValueError("w2c must contain sixteen finite values")
    else:
        rotation = torch.as_tensor(camera_data["R"])
        translation = torch.as_tensor(camera_data["T"])
        if rotation.numel() != 9 or translation.numel() != 3:
            raise ValueError("R and T must contain 9 and 3 values")
        if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
            raise ValueError("R and T must be finite")


def build_mmlphuman_camera(
    camera_data: dict[str, Any],
    image_height: int,
    image_width: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    """Build the camera dictionary consumed by ``GaussianModel.render``.

    MMLPHuman stores a conventional world-to-camera matrix ``[R|T]`` and uses
    it directly in gsplat. This adapter performs no axis flips or coordinate
    conversion; input calibration must already follow that convention.
    """

    validate_camera_data(camera_data)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image height and width must be positive")
    dtype = torch.float32
    intrinsic = torch.as_tensor(
        camera_data.get("K", camera_data.get("intrinsics")),
        dtype=dtype,
        device=device,
    ).reshape(3, 3)
    if "w2c" in camera_data:
        w2c = torch.as_tensor(camera_data["w2c"], dtype=dtype, device=device).reshape(4, 4)
    else:
        rotation = torch.as_tensor(camera_data["R"], dtype=dtype, device=device).reshape(3, 3)
        translation = torch.as_tensor(camera_data["T"], dtype=dtype, device=device).reshape(3)
        w2c = torch.eye(4, dtype=dtype, device=device)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation
    if not torch.allclose(
        w2c[3],
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        atol=1e-6,
    ):
        raise ValueError("w2c last row must be [0,0,0,1]")
    return {
        "K": intrinsic,
        "w2c": w2c,
        "height": int(image_height),
        "width": int(image_width),
    }
