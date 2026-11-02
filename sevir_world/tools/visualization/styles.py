"""Reusable visualization styles, including the canonical encoded-VIL palette.

The colors and bin edges match CasCast's ``datasets/sevir_util/sevir_cmap.py``,
which is adapted from the MIT-licensed SEVIR visualization utilities.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


CASCAST_VIL_LEVELS = np.asarray(
    [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0],
    dtype=np.float32,
)

# One color for each interval between adjacent levels.  RGB values are the
# exact 8-bit equivalents of CasCast's floating-point VIL_COLORS after its
# leading black missing-value color is removed.
CASCAST_VIL_COLORS_RGB = np.asarray(
    [
        [77, 77, 77],
        [40, 190, 40],
        [25, 150, 25],
        [10, 105, 10],
        [10, 75, 10],
        [245, 245, 0],
        [237, 172, 0],
        [240, 110, 0],
        [160, 0, 0],
        [231, 0, 255],
    ],
    dtype=np.uint8,
)
CASCAST_VIL_MISSING_RGB = np.asarray([0, 0, 0], dtype=np.uint8)


def matplotlib_style(name: str) -> tuple[ListedColormap, BoundaryNorm]:
    """Return a registered Matplotlib style without inspecting data provenance."""
    if name != "encoded_vil":
        raise ValueError(f"unknown visualization style: {name}")
    colors = CASCAST_VIL_COLORS_RGB.astype(np.float32) / 255.0
    cmap = ListedColormap(colors, name="cascast_sevir_vil")
    cmap.set_bad(CASCAST_VIL_MISSING_RGB.astype(np.float32) / 255.0)
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap, BoundaryNorm(CASCAST_VIL_LEVELS, cmap.N)


def mask_missing(values: np.ndarray, missing_value: float | int | None) -> np.ma.MaskedArray:
    """Apply the missing-value contract supplied by DataRef metadata."""
    array = np.asarray(values, dtype=np.float32)
    mask = ~np.isfinite(array)
    if missing_value is not None:
        mask |= array == float(missing_value)
    return np.ma.masked_where(mask, array)


def colorize(
    values: np.ndarray,
    *,
    style: str | None = None,
    missing_value: float | int | None = None,
) -> np.ndarray:
    """Colorize a 2-D grid from explicit style and variable metadata."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("visualization requires a 2-D grid")
    if style is None:
        finite_mask = np.isfinite(array)
        if missing_value is not None:
            finite_mask &= array != float(missing_value)
        finite = array[finite_mask]
        if finite.size:
            lower, upper = np.percentile(finite, [2, 98])
            scale = max(float(upper - lower), 1e-6)
            normalized = np.clip((array - lower) / scale, 0.0, 1.0)
        else:
            normalized = np.zeros_like(array)
        rgb = np.stack(
            [
                normalized * 255,
                (1.0 - np.abs(normalized - 0.5) * 2.0) * 210,
                (1.0 - normalized) * 255,
            ],
            axis=-1,
        ).astype(np.uint8)
        rgb[~finite_mask] = 0
        return rgb
    if style != "encoded_vil":
        raise ValueError(f"unknown visualization style: {style}")
    clipped = np.clip(array, 0.0, 254.999)
    clipped = np.where(np.isfinite(clipped), clipped, 0.0)
    indices = np.searchsorted(CASCAST_VIL_LEVELS[1:-1], clipped, side="right")
    rgb = CASCAST_VIL_COLORS_RGB[indices].copy()
    missing = ~np.isfinite(array)
    if missing_value is not None:
        missing |= array == float(missing_value)
    rgb[missing] = CASCAST_VIL_MISSING_RGB
    return rgb
