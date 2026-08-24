"""Canonical CasCast/SEVIR palette for encoded VIL displays.

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


def cascast_vil_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """Return the discrete Matplotlib colormap and norm used by CasCast."""
    colors = CASCAST_VIL_COLORS_RGB.astype(np.float32) / 255.0
    cmap = ListedColormap(colors, name="cascast_sevir_vil")
    cmap.set_bad(CASCAST_VIL_MISSING_RGB.astype(np.float32) / 255.0)
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap, BoundaryNorm(CASCAST_VIL_LEVELS, cmap.N)


def mask_missing_vil(values: np.ndarray) -> np.ma.MaskedArray:
    """Mask only SEVIR's encoded missing value while preserving predictions."""
    array = np.asarray(values, dtype=np.float32)
    return np.ma.masked_where(array == 255.0, array)


def colorize_vil(values: np.ndarray) -> np.ndarray:
    """Convert a 2-D encoded VIL grid to CasCast-compatible uint8 RGB."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("VIL colorization requires a 2-D grid")
    clipped = np.clip(array, 0.0, 254.999)
    clipped = np.where(np.isfinite(clipped), clipped, 0.0)
    indices = np.searchsorted(CASCAST_VIL_LEVELS[1:-1], clipped, side="right")
    rgb = CASCAST_VIL_COLORS_RGB[indices].copy()
    missing = ~np.isfinite(array) | (array == 255.0)
    rgb[missing] = CASCAST_VIL_MISSING_RGB
    return rgb
