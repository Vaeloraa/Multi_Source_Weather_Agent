from __future__ import annotations

"""Deterministic, interpretable signatures for past SEVIR VIL evolution."""

from datetime import datetime
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .contracts import AnalogueContractError, EventSignature


SIGNATURE_PIPELINE_VERSION = "sevir-vil-signature-extractor-1.0"
DEFAULT_WINDOW_MINUTES = (-60, 0)
DEFAULT_THRESHOLD = 133.0

_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("target_mean", "intensity", "encoded_vil"),
    ("target_q90", "intensity", "encoded_vil"),
    ("target_q99", "intensity", "encoded_vil"),
    ("sequence_mean", "intensity", "encoded_vil"),
    ("sequence_q90", "intensity", "encoded_vil"),
    ("sequence_max", "intensity", "encoded_vil"),
    ("coverage_74_target", "area", "fraction"),
    ("coverage_133_target", "area", "fraction"),
    ("coverage_74_max", "area", "fraction"),
    ("coverage_133_max", "area", "fraction"),
    ("coverage_133_growth", "temporal", "fraction"),
    ("mean_growth", "temporal", "encoded_vil"),
    ("mean_slope_per_frame", "temporal", "encoded_vil_per_frame"),
    ("coverage_133_slope_per_frame", "temporal", "fraction_per_frame"),
    ("intensifying_step_fraction", "temporal", "fraction"),
    ("mean_curve_0", "temporal_curve", "encoded_vil"),
    ("mean_curve_1", "temporal_curve", "encoded_vil"),
    ("mean_curve_2", "temporal_curve", "encoded_vil"),
    ("mean_curve_3", "temporal_curve", "encoded_vil"),
    ("mean_curve_4", "temporal_curve", "encoded_vil"),
    ("coverage_curve_0", "temporal_curve", "fraction"),
    ("coverage_curve_1", "temporal_curve", "fraction"),
    ("coverage_curve_2", "temporal_curve", "fraction"),
    ("coverage_curve_3", "temporal_curve", "fraction"),
    ("coverage_curve_4", "temporal_curve", "fraction"),
    ("centroid_dx", "motion", "normalized_grid"),
    ("centroid_dy", "motion", "normalized_grid"),
    ("centroid_displacement", "motion", "normalized_grid"),
    ("centroid_path_length", "motion", "normalized_grid"),
    ("centroid_speed_per_frame", "motion", "normalized_grid_per_frame"),
    ("object_count_target", "objects", "count"),
    ("largest_object_fraction_target", "objects", "fraction"),
    ("largest_object_compactness_target", "objects", "ratio"),
    ("largest_object_aspect_target", "objects", "ratio"),
    ("season_sin", "season", "unitless"),
    ("season_cos", "season", "unitless"),
    ("latitude_center", "geography", "degree_north"),
    ("longitude_center", "geography", "degree_east"),
    ("valid_fraction", "quality", "fraction"),
    ("active_frame_fraction", "quality", "fraction"),
)

FEATURE_NAMES = tuple(item[0] for item in _FEATURES)
FEATURE_GROUPS = tuple(item[1] for item in _FEATURES)
FEATURE_UNITS = tuple(item[2] for item in _FEATURES)


def _finite_stat(values: np.ndarray, function: Any) -> float | None:
    finite = values[np.isfinite(values)]
    return float(function(finite)) if finite.size else None


def _slope(values: np.ndarray) -> float | None:
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return None
    x = np.arange(values.size, dtype=np.float64)[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def _sample_curve(values: np.ndarray, count: int = 5) -> list[float | None]:
    mask = np.isfinite(values)
    if not np.any(mask):
        return [None] * count
    source_x = np.arange(values.size, dtype=np.float64)[mask]
    target_x = np.linspace(0.0, float(values.size - 1), count)
    return [float(item) for item in np.interp(target_x, source_x, values[mask])]


def _centroid(frame: np.ndarray, threshold: float = 74.0) -> tuple[float, float] | None:
    mask = np.isfinite(frame) & (frame >= threshold)
    if not np.any(mask):
        return None
    rows, columns = np.nonzero(mask)
    weights = np.maximum(frame[mask] - threshold + 1.0, 1.0)
    return (
        float(np.average(rows, weights=weights) / max(1, frame.shape[0] - 1)),
        float(np.average(columns, weights=weights) / max(1, frame.shape[1] - 1)),
    )


def _motion(frames: np.ndarray) -> dict[str, float | None]:
    points = [_centroid(frame) for frame in frames]
    valid = [(index, point) for index, point in enumerate(points) if point is not None]
    if len(valid) < 2:
        return {name: None for name in (
            "centroid_dx", "centroid_dy", "centroid_displacement",
            "centroid_path_length", "centroid_speed_per_frame",
        )}
    first_index, first = valid[0]
    last_index, last = valid[-1]
    assert first is not None and last is not None
    dy = float(last[0] - first[0])
    dx = float(last[1] - first[1])
    path = 0.0
    for (_, before), (_, after) in zip(valid, valid[1:]):
        assert before is not None and after is not None
        path += math.hypot(after[0] - before[0], after[1] - before[1])
    span = max(1, last_index - first_index)
    return {
        "centroid_dx": dx,
        "centroid_dy": dy,
        "centroid_displacement": float(math.hypot(dx, dy)),
        "centroid_path_length": float(path),
        "centroid_speed_per_frame": float(path / span),
    }


def _objects(frame: np.ndarray, threshold: float) -> dict[str, float]:
    mask = np.isfinite(frame) & (frame >= threshold)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return {
            "object_count_target": 0.0,
            "largest_object_fraction_target": 0.0,
            "largest_object_compactness_target": 0.0,
            "largest_object_aspect_target": 0.0,
        }
    sizes = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(sizes) + 1)
    largest = labels == largest_label
    area = int(sizes[largest_label - 1])
    rows, columns = np.nonzero(largest)
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)
    eroded = ndimage.binary_erosion(largest, structure=np.ones((3, 3), dtype=bool))
    perimeter = int(np.count_nonzero(largest & ~eroded))
    compactness = min(1.0, float(4.0 * math.pi * area / max(1, perimeter ** 2)))
    return {
        "object_count_target": float(count),
        "largest_object_fraction_target": float(area / frame.size),
        "largest_object_compactness_target": compactness,
        "largest_object_aspect_target": float(max(height, width) / max(1, min(height, width))),
    }


def _month(context: Mapping[str, Any], anchor: str) -> int | None:
    raw = context.get("month")
    if raw is not None:
        value = int(raw)
        return value if 1 <= value <= 12 else None
    if anchor:
        try:
            return datetime.fromisoformat(anchor.replace("Z", "+00:00")).month
        except ValueError:
            try:
                return datetime.strptime(anchor, "%Y-%m-%d %H:%M:%S").month
            except ValueError:
                return None
    return None


def _geo(context: Mapping[str, Any]) -> tuple[float | None, float | None]:
    metadata = context.get("spatial_metadata")
    bounds = metadata.get("bounds") if isinstance(metadata, Mapping) else context.get("bounding_box")
    if not isinstance(bounds, Mapping):
        return None, None
    try:
        return (
            (float(bounds["south"]) + float(bounds["north"])) / 2.0,
            (float(bounds["west"]) + float(bounds["east"])) / 2.0,
        )
    except (KeyError, TypeError, ValueError):
        return None, None


def _select_past_window(
    frames: np.ndarray,
    minute_offsets: Sequence[int] | None,
    window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if frames.ndim != 3:
        raise AnalogueContractError("VIL frames must have dimensions time,row,column")
    if minute_offsets is None:
        if frames.shape[0] == 49:
            offsets = np.arange(-120, 125, 5, dtype=np.int32)
            offset_source = "canonical_sevir_49_frame_axis"
        elif frames.shape[0] == 13:
            offsets = np.arange(-60, 5, 5, dtype=np.int32)
            offset_source = "canonical_cascast_13_frame_history_axis"
        else:
            raise AnalogueContractError(
                "minute_offsets are required for non-canonical frame counts"
            )
    else:
        offsets = np.asarray(tuple(int(item) for item in minute_offsets), dtype=np.int32)
        offset_source = "provided_minute_offsets"
    if offsets.size != frames.shape[0]:
        raise AnalogueContractError("minute_offsets length does not match frame count")
    start, end = (int(window[0]), int(window[1]))
    if end > 0 or start > end:
        raise AnalogueContractError("signature window must end at or before minute zero")
    positions = np.flatnonzero((offsets >= start) & (offsets <= end))
    if positions.size < 2:
        raise AnalogueContractError("fewer than two frames fall inside the signature window")
    return frames[positions], offsets[positions], [offset_source, *positions.astype(int).tolist()]


def extract_event_signature(
    frames: Any,
    *,
    subject_key: str,
    event_id: str,
    source: str,
    source_fingerprint: Mapping[str, Any],
    minute_offsets: Sequence[int] | None = None,
    window: tuple[int, int] = DEFAULT_WINDOW_MINUTES,
    threshold: float = DEFAULT_THRESHOLD,
    event_anchor_time_utc: str = "",
    event_type: str = "Unlabeled",
    context: Mapping[str, Any] | None = None,
) -> EventSignature:
    """Extract a fixed, interpretable signature using only times at or before 0.

    The caller may pass a complete 49-frame event.  Frames after minute zero
    are selected out before any statistic is computed, making future leakage a
    testable contract rather than a prompt instruction.
    """

    raw = np.asarray(frames)
    selected, selected_offsets, selection = _select_past_window(raw, minute_offsets, window)
    values = selected.astype(np.float64, copy=True)
    values[(values == 255.0) | ~np.isfinite(values)] = np.nan
    frame_means = np.asarray([
        _finite_stat(frame, np.mean) for frame in values
    ], dtype=np.float64)
    coverage_74 = np.asarray([
        float(np.count_nonzero(np.isfinite(frame) & (frame >= 74.0)) / np.count_nonzero(np.isfinite(frame)))
        if np.any(np.isfinite(frame)) else np.nan
        for frame in values
    ])
    coverage_133 = np.asarray([
        float(np.count_nonzero(np.isfinite(frame) & (frame >= threshold)) / np.count_nonzero(np.isfinite(frame)))
        if np.any(np.isfinite(frame)) else np.nan
        for frame in values
    ])
    context_value = dict(context or {})
    month = _month(context_value, event_anchor_time_utc)
    latitude, longitude = _geo(context_value)
    season_angle = 2.0 * math.pi * (month - 1) / 12.0 if month is not None else None
    target = values[-1]
    finite_target = target[np.isfinite(target)]
    finite_all = values[np.isfinite(values)]
    feature_map: dict[str, float | None] = {
        "target_mean": float(np.mean(finite_target)) if finite_target.size else None,
        "target_q90": float(np.quantile(finite_target, 0.90)) if finite_target.size else None,
        "target_q99": float(np.quantile(finite_target, 0.99)) if finite_target.size else None,
        "sequence_mean": float(np.mean(finite_all)) if finite_all.size else None,
        "sequence_q90": float(np.quantile(finite_all, 0.90)) if finite_all.size else None,
        "sequence_max": float(np.max(finite_all)) if finite_all.size else None,
        "coverage_74_target": float(coverage_74[-1]) if np.isfinite(coverage_74[-1]) else None,
        "coverage_133_target": float(coverage_133[-1]) if np.isfinite(coverage_133[-1]) else None,
        "coverage_74_max": _finite_stat(coverage_74, np.max),
        "coverage_133_max": _finite_stat(coverage_133, np.max),
        "coverage_133_growth": (
            float(coverage_133[-1] - coverage_133[0])
            if np.isfinite(coverage_133[[0, -1]]).all() else None
        ),
        "mean_growth": (
            float(frame_means[-1] - frame_means[0])
            if np.isfinite(frame_means[[0, -1]]).all() else None
        ),
        "mean_slope_per_frame": _slope(frame_means),
        "coverage_133_slope_per_frame": _slope(coverage_133),
        "intensifying_step_fraction": (
            float(np.mean(np.diff(frame_means) > 0))
            if frame_means.size > 1 and np.isfinite(frame_means).all() else None
        ),
        **_motion(values),
        **_objects(target, float(threshold)),
        "season_sin": float(math.sin(season_angle)) if season_angle is not None else None,
        "season_cos": float(math.cos(season_angle)) if season_angle is not None else None,
        "latitude_center": latitude,
        "longitude_center": longitude,
        "valid_fraction": float(np.count_nonzero(np.isfinite(values)) / values.size),
        "active_frame_fraction": float(np.mean(coverage_133 > 0)),
    }
    for index, item in enumerate(_sample_curve(frame_means)):
        feature_map[f"mean_curve_{index}"] = item
    for index, item in enumerate(_sample_curve(coverage_133)):
        feature_map[f"coverage_curve_{index}"] = item
    features = tuple(feature_map[name] for name in FEATURE_NAMES)
    return EventSignature(
        subject_key=subject_key,
        event_id=event_id,
        event_anchor_time_utc=event_anchor_time_utc,
        event_type=event_type or "Unlabeled",
        feature_names=FEATURE_NAMES,
        features=features,
        feature_units=FEATURE_UNITS,
        feature_groups=FEATURE_GROUPS,
        frame_count=int(values.shape[0]),
        window_start_minute=int(selected_offsets[0]),
        window_end_minute=int(selected_offsets[-1]),
        threshold=float(threshold),
        source=source,
        source_fingerprint=source_fingerprint,
        context={**context_value, "month": month},
        quality={
            "valid_fraction": feature_map["valid_fraction"],
            "active_frame_fraction": feature_map["active_frame_fraction"],
            "finite_feature_count": int(sum(
                bool(item is not None and np.isfinite(item)) for item in features
            )),
            "feature_count": len(features),
        },
        extraction={
            "pipeline_version": SIGNATURE_PIPELINE_VERSION,
            "selected_offsets": selected_offsets.astype(int).tolist(),
            "offset_source": selection[0],
            "selected_source_positions": selection[1:],
            "future_frames_read_by_signature": False,
            "missing_value": 255,
            "thresholds": [74.0, float(threshold)],
        },
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW_MINUTES",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "SIGNATURE_PIPELINE_VERSION",
    "extract_event_signature",
]
