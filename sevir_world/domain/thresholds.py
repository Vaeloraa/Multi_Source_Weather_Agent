"""Canonical CasCast VIL threshold gradient.

CasCast evaluates SEVIR VIL at these encoded 0--255 cut points.  They are
analysis/evaluation criteria, not calibrated rainfall, dBZ, hail, or warning
categories.
"""

from __future__ import annotations

CASCAST_VIL_THRESHOLDS: tuple[float, ...] = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)
DEFAULT_CASCAST_VIL_THRESHOLD: float = 133.0


def normalize_cascast_vil_threshold(value: object) -> float:
    """Return an exact CasCast cut point, rejecting ad-hoc thresholds."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("VIL threshold must be one of the CasCast gradient values") from error
    if threshold not in CASCAST_VIL_THRESHOLDS:
        values = ", ".join(str(int(item)) for item in CASCAST_VIL_THRESHOLDS)
        raise ValueError(f"VIL threshold must be one of [{values}]")
    return threshold


def cascast_vil_threshold_payload() -> list[int]:
    return [int(item) for item in CASCAST_VIL_THRESHOLDS]
