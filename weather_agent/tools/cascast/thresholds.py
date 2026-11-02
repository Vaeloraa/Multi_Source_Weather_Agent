"""CasCast encoded-VIL threshold validation and evaluation cut points.

CasCast evaluates SEVIR VIL at these encoded 0--255 cut points.  They are
analysis/evaluation criteria, not calibrated rainfall, dBZ, hail, or warning
categories.
"""

from __future__ import annotations

CASCAST_VIL_THRESHOLDS: tuple[float, ...] = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)
DEFAULT_CASCAST_VIL_THRESHOLD: float = 133.0


def normalize_cascast_vil_threshold(value: object) -> float:
    """Validate an explicitly selected encoded-VIL analysis threshold.

    The canonical cut points remain available for evaluation summaries and
    palette controls, but analysis code may select any finite encoded VIL
    value from 0 through 254. Encoded value 255 denotes missing data.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("VIL threshold must be a finite encoded value from 0 through 254") from error
    if not 0.0 <= threshold <= 254.0:
        raise ValueError("VIL threshold must be a finite encoded value from 0 through 254")
    return threshold


def cascast_vil_threshold_payload() -> list[int]:
    return [int(item) for item in CASCAST_VIL_THRESHOLDS]
