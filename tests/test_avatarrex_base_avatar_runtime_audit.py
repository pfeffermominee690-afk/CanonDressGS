import numpy as np

from tools.datasets.audit_avatarrex_base_avatar_preflight import (
    _project,
    _unproject,
    audit_cameras,
)


def _synthetic_calibration() -> dict[str, dict[str, object]]:
    calibration = {}
    for index in range(16):
        angle = 2.0 * np.pi * index / 16.0
        rotation = np.eye(3, dtype=np.float64)
        center = np.array([np.sin(angle), 0.1 * np.cos(angle), np.cos(angle)], dtype=np.float64)
        translation = -(rotation @ center)
        calibration[f"{22000000 + index}"] = {
            "K": [[1250.0, 0.0, 750.0], [0.0, 1250.0, 1024.0], [0.0, 0.0, 1.0]],
            "R": rotation.tolist(),
            "T": translation.tolist(),
            "distCoeff": [0.0] * 5,
            "imgSize": [1500, 2048],
            "rectifyAlpha": 0.0,
        }
    return calibration


def test_projection_roundtrip_uses_world_to_camera_convention() -> None:
    K = np.array([[1200.0, 0.0, 750.0], [0.0, 1210.0, 1024.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    T = np.array([0.2, -0.1, 0.3])
    pixels = np.array([[750.0, 1024.0], [375.0, 512.0], [1125.0, 1536.0]])
    world = _unproject(K, R, T, pixels, depth=2.0)
    projected, depths = _project(K, R, T, world)

    np.testing.assert_allclose(projected, pixels, atol=1e-10)
    np.testing.assert_allclose(depths, np.full(3, 2.0), atol=1e-10)


def test_fixed_sixteen_camera_contract() -> None:
    result = audit_cameras(_synthetic_calibration())

    assert result["status"] == "PASS"
    assert result["camera_count"] == 16
    assert result["imgSize_order"] == "width_height"
    assert result["distortion_policy"]["all_coefficients_zero"] is True
    assert all(row["status"] == "PASS" for row in result["rows"])
