"""Pure, side-effect-free reference primitives for the repaired P0 protocol.

This module transforms in-memory arrays only.  It does not load a model, render,
evaluate, train, create an optimizer, or write an artifact.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil, floor, sqrt
from typing import Sequence

import numpy as np


SOURCE_HEAD = "32e6058ea64078f5ccded6f86e8bffe3cb81b30e"
OUTFIT_ORDER = ("O01", "O02", "O03", "O04", "O08")
TARGET_CONDITION_ORDER = (
    "cond_000000",
    "cond_000318",
    "cond_000017",
    "cond_000347",
)
PAIR_ORDER = tuple(combinations(OUTFIT_ORDER, 2))

C3_QUANTILE_COUNT = 256
C3_QUANTILES = np.linspace(0.0, 1.0, C3_QUANTILE_COUNT, dtype=np.float64)
C4_EPSILON = 1.0e-6
C5_KERNEL_SIZE = 11
C5_SIGMA = 3.0

GRAYSCALE_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)
HUE_LADDER_DEGREES = (0, 15, 30, 45, 60)
BLUR_LADDER_SIGMA = (0, 1, 2, 3, 4)
MASK_MORPHOLOGY_RADIUS = (-8, -4, 0, 4, 8)
INTERPOLATION_ALPHAS = tuple(index / 10.0 for index in range(11))


def _require_rgb_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(rgb)
    membership = np.asarray(mask)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[-1] != 3:
        raise TypeError("RGB must be float32 HxWx3")
    if membership.dtype != np.bool_ or membership.shape != value.shape[:2]:
        raise TypeError("mask must be bool HxW and match RGB")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("RGB must be finite and in [0,1]")
    if not membership.any():
        raise ValueError("clothing mask must contain at least one pixel")
    return value, membership


def round_half_up_8bit(value: np.ndarray) -> np.ndarray:
    """Quantize clipped [0,1] values with floor(255*x+0.5)/255."""

    array = np.asarray(value, dtype=np.float64)
    return (np.floor(255.0 * np.clip(array, 0.0, 1.0) + 0.5) / 255.0).astype(
        np.float32
    )


def _strict_quantile_knots(
    source_quantiles: np.ndarray, destination_quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse equal source knots by the mean destination ordinate.

    Empirical 8-bit RGB commonly creates repeated source quantiles.  Averaging
    the destination ordinates makes the piecewise-linear map single-valued and
    deterministic without adaptive binning.
    """

    unique, inverse = np.unique(source_quantiles, return_inverse=True)
    sums = np.zeros(unique.shape, dtype=np.float64)
    counts = np.zeros(unique.shape, dtype=np.int64)
    np.add.at(sums, inverse, destination_quantiles)
    np.add.at(counts, inverse, 1)
    return unique, sums / counts


def c3_histogram_match(
    source_rgb: np.ndarray,
    destination_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    destination_clothing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the repaired C3 mapping to masked RGB pixels only."""

    source, mask = _require_rgb_mask(source_rgb, clothing_mask)
    donor_mask = clothing_mask if destination_clothing_mask is None else destination_clothing_mask
    destination, destination_mask = _require_rgb_mask(destination_rgb, donor_mask)
    output = source.copy()
    for channel in range(3):
        source_values = source[..., channel][mask].astype(np.float64)
        destination_values = destination[..., channel][destination_mask].astype(np.float64)
        source_q = np.quantile(source_values, C3_QUANTILES, method="linear")
        destination_q = np.quantile(
            destination_values, C3_QUANTILES, method="linear"
        )
        knots_x, knots_y = _strict_quantile_knots(source_q, destination_q)
        if len(knots_x) == 1:
            mapped = np.full(source_values.shape, knots_y[0], dtype=np.float64)
        else:
            mapped = np.interp(source_values, knots_x, knots_y)
        channel_output = output[..., channel]
        channel_output[mask] = round_half_up_8bit(mapped)
    return output


def c4_brightness_contrast_normalize(
    source_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    target_mean: Sequence[float],
    target_std: Sequence[float],
) -> np.ndarray:
    """Map masked channel statistics to frozen global reference statistics."""

    source, mask = _require_rgb_mask(source_rgb, clothing_mask)
    wanted_mean = np.asarray(target_mean, dtype=np.float64)
    wanted_std = np.asarray(target_std, dtype=np.float64)
    if wanted_mean.shape != (3,) or wanted_std.shape != (3,):
        raise ValueError("target mean/std must each contain three channels")
    if not np.isfinite(wanted_mean).all() or not np.isfinite(wanted_std).all():
        raise ValueError("target statistics must be finite")
    if np.any(wanted_std < 0.0):
        raise ValueError("target standard deviation must be nonnegative")

    output = source.copy()
    for channel in range(3):
        values = source[..., channel][mask].astype(np.float64)
        source_mean = values.mean(dtype=np.float64)
        source_variance = np.mean(
            np.square(values - source_mean), dtype=np.float64
        )
        source_std = sqrt(float(source_variance))
        if source_std < C4_EPSILON:
            mapped = np.full(values.shape, wanted_mean[channel], dtype=np.float64)
        else:
            mapped = (
                (values - source_mean) / max(source_std, C4_EPSILON)
            ) * wanted_std[channel] + wanted_mean[channel]
        output[..., channel][mask] = np.clip(mapped, 0.0, 1.0).astype(np.float32)
    return output


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-np.square(offsets) / (2.0 * sigma * sigma))
    kernel /= kernel.sum(dtype=np.float64)
    return kernel.astype(np.float32)


def _separable_reflect_blur(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    if rgb.shape[0] <= radius or rgb.shape[1] <= radius:
        raise ValueError("reflect padding requires both dimensions to exceed radius")
    padded_y = np.pad(rgb, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    vertical = np.zeros_like(rgb, dtype=np.float32)
    for index, weight in enumerate(kernel):
        vertical += np.float32(weight) * padded_y[index : index + rgb.shape[0]]
    padded_x = np.pad(vertical, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    horizontal = np.zeros_like(rgb, dtype=np.float32)
    for index, weight in enumerate(kernel):
        horizontal += np.float32(weight) * padded_x[:, index : index + rgb.shape[1]]
    return horizontal


def c5_gaussian_blur(source_rgb: np.ndarray, clothing_mask: np.ndarray) -> np.ndarray:
    """Blur already-resized RGB, then restore every pixel outside the mask."""

    source, mask = _require_rgb_mask(source_rgb, clothing_mask)
    kernel = gaussian_kernel_1d(C5_KERNEL_SIZE, C5_SIGMA)
    blurred = np.clip(_separable_reflect_blur(source, kernel), 0.0, 1.0).astype(
        np.float32
    )
    output = source.copy()
    output[mask] = blurred[mask]
    return output


def fixed_grayscale(source_rgb: np.ndarray) -> np.ndarray:
    """Apply the frozen sRGB-numeric luminance conversion to every pixel."""

    source = np.asarray(source_rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3:
        raise TypeError("RGB must be float32 HxWx3")
    if not np.isfinite(source).all() or np.any(source < 0.0) or np.any(source > 1.0):
        raise ValueError("RGB must be finite and in [0,1]")
    luminance = (
        source[..., 0].astype(np.float64) * 0.299
        + source[..., 1].astype(np.float64) * 0.587
        + source[..., 2].astype(np.float64) * 0.114
    )
    return np.repeat(np.clip(luminance, 0.0, 1.0)[..., None], 3, axis=-1).astype(
        np.float32
    )


def hue_shift(source_rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Shift standard HSV hue over the whole sRGB-numeric image."""

    source = np.asarray(source_rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3:
        raise TypeError("RGB must be float32 HxWx3")
    if not np.isfinite(source).all() or np.any(source < 0.0) or np.any(source > 1.0):
        raise ValueError("RGB must be finite and in [0,1]")
    rgb = source.astype(np.float64)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    hue = np.zeros(maximum.shape, dtype=np.float64)
    nonzero = delta > 0.0
    red = nonzero & (maximum == rgb[..., 0])
    green = nonzero & (maximum == rgb[..., 1])
    blue = nonzero & (maximum == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    hue = ((hue / 6.0) + float(degrees) / 360.0) % 1.0
    saturation = np.divide(
        delta, maximum, out=np.zeros_like(delta), where=maximum > 0.0
    )
    sector = hue * 6.0
    index = np.floor(sector).astype(np.int64) % 6
    fraction = sector - np.floor(sector)
    p = maximum * (1.0 - saturation)
    q = maximum * (1.0 - saturation * fraction)
    t = maximum * (1.0 - saturation * (1.0 - fraction))
    values = (
        (maximum, t, p), (q, maximum, p), (p, maximum, t),
        (p, q, maximum), (t, p, maximum), (maximum, p, q),
    )
    output = np.empty_like(rgb)
    for sector_index, channels in enumerate(values):
        membership = index == sector_index
        for channel, value in enumerate(channels):
            output[..., channel][membership] = value[membership]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def c6_average_garment_color(
    source_rgb: np.ndarray, clothing_mask: np.ndarray
) -> np.ndarray:
    """Fill the exact frozen mask interior with its float64 channel mean."""

    source, mask = _require_rgb_mask(source_rgb, clothing_mask)
    mean = source[mask].astype(np.float64).mean(axis=0, dtype=np.float64)
    output = source.copy()
    output[mask] = np.clip(mean, 0.0, 1.0).astype(np.float32)
    return output


def grayscale_ladder(source_rgb: np.ndarray, value: float) -> np.ndarray:
    if value < 0.0 or value > 1.0:
        raise ValueError("grayscale ladder value must be in [0,1]")
    source = np.asarray(source_rgb, dtype=np.float32)
    result = (1.0 - np.float32(value)) * source + np.float32(value) * fixed_grayscale(source)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def blur_ladder(source_rgb: np.ndarray, sigma: float) -> np.ndarray:
    source = np.asarray(source_rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3:
        raise TypeError("RGB must be float32 HxWx3")
    if sigma == 0.0:
        return source.copy()
    kernel = gaussian_kernel_1d(blur_ladder_kernel_size(float(sigma)), float(sigma))
    return np.clip(_separable_reflect_blur(source, kernel), 0.0, 1.0).astype(np.float32)


def disk_morphology(mask: np.ndarray, radius: int) -> np.ndarray:
    """Apply the frozen integer-disk binary erosion/dilation rule."""

    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.ndim != 2:
        raise TypeError("mask must be bool HxW")
    if not isinstance(radius, int) or isinstance(radius, bool):
        raise TypeError("radius must be an integer")
    if radius == 0:
        return value.copy()
    magnitude = abs(radius)
    padded = np.pad(
        value,
        ((magnitude, magnitude), (magnitude, magnitude)),
        mode="constant",
        constant_values=False,
    )
    shifted = []
    height, width = value.shape
    for dy in range(-magnitude, magnitude + 1):
        for dx in range(-magnitude, magnitude + 1):
            if dx * dx + dy * dy <= magnitude * magnitude:
                shifted.append(
                    padded[
                        magnitude + dy : magnitude + dy + height,
                        magnitude + dx : magnitude + dx + width,
                    ]
                )
    stack = np.stack(shifted, axis=0)
    return stack.all(axis=0) if radius < 0 else stack.any(axis=0)


def mixed_reference_assignments() -> tuple[tuple[str, tuple[str, str, str]], ...]:
    return (
        ("AAA", ("A", "A", "A")),
        ("AAB_B_POSITION_0", ("B", "A", "A")),
        ("AAB_B_POSITION_1", ("A", "B", "A")),
        ("AAB_B_POSITION_2", ("A", "A", "B")),
        ("ABB_A_POSITION_0", ("A", "B", "B")),
        ("ABB_A_POSITION_1", ("B", "A", "B")),
        ("ABB_A_POSITION_2", ("B", "B", "A")),
        ("BBB", ("B", "B", "B")),
    )


def mixed_reference_query_count_per_method() -> int:
    return len(PAIR_ORDER) * len(TARGET_CONDITION_ORDER) * len(
        mixed_reference_assignments()
    )


def interpolation_grid_size() -> int:
    return len(PAIR_ORDER) * len(INTERPOLATION_ALPHAS) * len(TARGET_CONDITION_ORDER)


def blur_ladder_kernel_size(sigma: float) -> int:
    if sigma < 0.0:
        raise ValueError("sigma cannot be negative")
    return 1 if sigma == 0.0 else 2 * ceil(3.0 * sigma) + 1


def trust_radii(b_xyz: float | Sequence[float]) -> tuple[float, float]:
    value = np.asarray(b_xyz, dtype=np.float64)
    if value.ndim == 0:
        tau_one = float(value)
    elif value.shape == (3,):
        tau_one = float(np.linalg.norm(value, ord=2))
    else:
        raise ValueError("b_xyz must be scalar or length-three")
    if not np.isfinite(tau_one) or tau_one <= 0.0:
        raise ValueError("b_xyz must define a finite positive radius")
    return 0.5 * tau_one, tau_one


def boundary_tolerance(height: int, width: int) -> int:
    if height <= 0 or width <= 0:
        raise ValueError("render dimensions must be positive")
    return max(1, floor(0.005 * min(height, width) + 0.5))
