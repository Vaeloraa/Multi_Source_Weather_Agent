from __future__ import annotations

import base64
import binascii
import csv
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..data_access.config import *
from ..data_access.catalog import (
    _catalog_public_record,
    _catalog_vil_event,
    _catalog_vil_records,
    _load_vil_array,
    _local_hdf5_path,
)
from ..data_access.ensemble import _load_cascast_ensemble
from . import geospatial

import h5py
import numpy as np
from pyproj import CRS, Transformer


def _comparison_metrics(frames: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute neutral, same-definition metrics for forecast/history frames."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    valid = values != 255
    clean = np.where(valid, values, np.nan)
    valid_count = np.sum(valid, axis=(1, 2))
    maxima = np.nanmax(clean, axis=(1, 2))
    means = np.nanmean(clean, axis=(1, 2))
    p95 = np.nanquantile(clean, 0.95, axis=(1, 2))
    coverage = np.sum((values >= float(threshold)) & valid, axis=(1, 2)) / np.maximum(valid_count, 1)

    def scalar_summary(series: np.ndarray) -> dict[str, float | None]:
        finite = np.asarray(series, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return {"max": None, "mean": None, "p95": None}
        return {
            "max": round(float(np.max(finite)), 4),
            "mean": round(float(np.mean(finite)), 4),
            "p95": round(float(np.quantile(finite, 0.95)), 4),
        }

    return {
        "frame_count": int(values.shape[0]),
        "threshold": float(threshold),
        "max_vil": scalar_summary(maxima),
        "mean_vil": scalar_summary(means),
        "p95_vil": scalar_summary(p95),
        "threshold_coverage": scalar_summary(coverage),
        "per_frame": [
            {
                "frame_index": index,
                "max_vil": round(float(maxima[index]), 4) if np.isfinite(maxima[index]) else None,
                "mean_vil": round(float(means[index]), 4) if np.isfinite(means[index]) else None,
                "p95_vil": round(float(p95[index]), 4) if np.isfinite(p95[index]) else None,
                "threshold_coverage": round(float(coverage[index]), 6),
                "valid_pixel_count": int(valid_count[index]),
            }
            for index in range(values.shape[0])
        ],
    }


def catalog_schema() -> dict[str, Any]:
    """Discover the fields and observed types from the loaded catalog schema."""
    records = _catalog_vil_records()
    fields: dict[str, dict[str, Any]] = {}
    for field in records[0].keys():
        values = [row.get(field) for row in records[:1000] if row.get(field) is not None]
        types = sorted({type(value).__name__ for value in values})
        fields[field] = {
            "types": types or ["null"],
            "numeric": bool(values) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values),
        }
    return {
        "status": "ok",
        "task": "catalog_schema",
        "capabilities": ["catalog_schema_discovery"],
        "provenance": _catalog_provenance(),
        "record_type": "SEVIR VIL catalog metadata",
        "fields": fields,
        "record_count": len(records),
    }


def _runtime_catalog_fields() -> set[str]:
    return set(catalog_schema()["fields"])


def _runtime_numeric_fields() -> set[str]:
    return {
        field for field, metadata in catalog_schema()["fields"].items()
        if metadata["numeric"]
    }
AGGREGATE_OPERATIONS = frozenset(
    {"count", "mean", "min", "max", "median", "std", "sum", "quantile"}
)


def _catalog_provenance(*, source_kind: str = "sevir_catalog") -> dict[str, Any]:
    """Return machine-readable provenance shared by low-level statistics APIs."""
    return {
        "source_kind": source_kind,
        "dataset": "SEVIR",
        "modality": "VIL",
        "value_space": "uint8 encoded VIL; 255 denotes missing data",
        "official_warning_product": False,
    }


def query_sevir_records(
    year: int | None = None,
    event_type: str = "",
    fields: list[str] | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Return a bounded, filterable view of SEVIR VIL catalog records.

    This is intentionally a data primitive rather than a precomputed report. The
    agent chooses fields and filters; aggregation is performed separately.
    """
    requested_fields = list(fields or ["id", "year", "month", "event_type", "data_max"])
    unknown = sorted(set(requested_fields) - _runtime_catalog_fields())
    if unknown:
        raise ValueError(f"unsupported catalog fields: {', '.join(unknown)}")
    if not requested_fields:
        raise ValueError("fields must contain at least one catalog field")
    bounded_limit = max(1, min(int(limit), 50000))
    normalized_type = str(event_type or "").strip().lower()
    records = [
        record
        for record in _catalog_vil_records()
        if (year is None or record["year"] == int(year))
        and (not normalized_type or normalized_type in str(record["event_type"]).lower())
    ]
    projected = [
        {field: record.get(field) for field in requested_fields}
        for record in records[:bounded_limit]
    ]
    return {
        "status": "ok",
        "task": "catalog_query",
        "capabilities": ["catalog_metadata", "agent_selected_fields"],
        "provenance": _catalog_provenance(),
        "filters": {"year": year, "event_type": event_type},
        "fields": requested_fields,
        "record_count": len(records),
        "returned_count": len(projected),
        "truncated": len(records) > bounded_limit,
        "records": projected,
    }


def query_sevir_values(
    field: str = "data_max",
    year: int | None = None,
    event_type: str = "",
    limit: int = 50000,
) -> dict[str, Any]:
    """Return a bounded numeric series for agent-written calculations.

    Unlike aggregate helpers, this function does not choose a statistic. It only
    applies the data filters and returns values so generated code can calculate the
    requested quantity with ordinary safe Python operations.
    """
    if field not in _runtime_numeric_fields():
        raise ValueError(f"unsupported catalog field: {field}")
    bounded_limit = max(1, min(int(limit), 50000))
    normalized_type = str(event_type or "").strip().lower()
    records = [
        record
        for record in _catalog_vil_records()
        if (year is None or record["year"] == int(year))
        and (not normalized_type or normalized_type in str(record["event_type"]).lower())
    ]
    values = [
        record[field]
        for record in records[:bounded_limit]
        if record.get(field) is not None
    ]
    return {
        "status": "ok",
        "task": "catalog_values",
        "capabilities": ["catalog_metadata", "numeric_series"],
        "provenance": _catalog_provenance(),
        "field": field,
        "filters": {"year": year, "event_type": event_type},
        "source_record_count": len(records),
        "valid_value_count": len(values),
        "truncated": len(records) > bounded_limit,
        "values": values,
    }


def find_similar_event_candidates(
    reference_event_id: str,
    field: str = "data_max",
    candidate_limit: int = 10,
    same_event_type: bool = True,
    local_hdf5_only: bool = True,
) -> dict[str, Any]:
    """Return metadata candidates under an explicit Agent-selected similarity rule.

    This primitive ranks catalog rows by distance on the requested numeric field;
    it does not read HDF5 or calculate duration.  The caller must subsequently
    load the returned IDs with ``load_event_history_frame_primitives`` and compute
    the duration/group statistic itself.
    """
    reference = _catalog_vil_event(str(reference_event_id).strip())
    if field not in _runtime_numeric_fields():
        raise ValueError(f"unsupported similarity field: {field}")
    limit = max(1, min(int(candidate_limit), 20))
    reference_value = reference.get(field)
    if reference_value is None:
        raise ValueError(f"reference event has no value for similarity field: {field}")
    catalog_rows = list(_catalog_vil_records())
    id_counts: dict[str, int] = {}
    for row in catalog_rows:
        id_counts[str(row.get("id", "")).lower()] = id_counts.get(
            str(row.get("id", "")).lower(), 0
        ) + 1
    rows = []
    unavailable_count = 0
    ambiguous_count = 0
    for row in catalog_rows:
        if row.get("id") == reference["id"]:
            continue
        if id_counts.get(str(row.get("id", "")).lower(), 0) != 1:
            ambiguous_count += 1
            continue
        local_available = bool(row.get("file_name")) and _local_hdf5_path(row).exists()
        if local_hdf5_only and not local_available:
            unavailable_count += 1
            continue
        if same_event_type and str(reference.get("event_type", "")).strip().lower() not in {"", "unlabeled"}:
            if str(row.get("event_type", "")).strip().lower() != str(reference.get("event_type", "")).strip().lower():
                continue
        value = row.get(field)
        if value is None:
            continue
        rows.append({
            "event_id": row["id"],
            "event_type": row.get("event_type"),
            field: value,
            "absolute_distance": abs(float(value) - float(reference_value)),
            "local_hdf5_available": bool(local_available),
        })
    rows.sort(key=lambda item: (item["absolute_distance"], item["event_id"]))
    selected = rows[:limit]
    selected_values = [item.get(field) for item in selected]
    zero_distance_count = sum(
        1 for item in selected if float(item["absolute_distance"]) == 0.0
    )
    nondiscriminating = bool(selected) and len(set(selected_values)) <= 1
    return {
        "status": "ok",
        "task": "similar_event_candidates",
        "capabilities": ["catalog_metadata", "candidate_discovery"],
        "provenance": _catalog_provenance(),
        "reference_event_id": reference["id"],
        "reference": {"event_type": reference.get("event_type"), field: reference_value},
        "records": selected,
        "candidate_count": len(selected),
        "similarity_definition": {
            "field": field,
            "rule": "nearest absolute numeric distance",
            "same_event_type": bool(same_event_type),
            "local_hdf5_only": bool(local_hdf5_only),
            "candidate_limit": limit,
            "duration_not_computed": True,
            "meteorological_similarity_established": False,
        },
        "interpretation_limit": (
            "Candidates are metadata-nearest events, not meteorologically equivalent storms. "
            "Read their HDF5 histories before making any duration statistic."
        ),
        "selection_diagnostics": {
            "unavailable_local_hdf5_rows_excluded": unavailable_count,
            "ambiguous_event_id_rows_excluded": ambiguous_count,
            "selected_unique_field_value_count": len(set(selected_values)),
            "selected_zero_distance_count": zero_distance_count,
            "selected_field_is_nondiscriminating": nondiscriminating,
        },
        "recommended_next_step": (
            "Load the reference event and candidate IDs together with "
            "statistics.event_histories_raw, derive question-relevant features, and "
            "let generated code make the final similarity comparison."
        ),
    }


def compare_cascast_with_historical_region(
    event_id: str,
    month_window: int = 1,
    day_window: int | None = 15,
    threshold: float = 40.0,
    history_frame_count: int = 12,
    candidate_limit: int = 10,
    region: list[int] | None = None,
) -> dict[str, Any]:
    """Compare a cached CasCast forecast with same-region historical VIL.

    ``region`` is an optional normalized pixel box ``[row0, col0, row1, col1]``.
    It keeps this primitive independent of a particular city gazetteer while
    allowing a geocoder/UI to translate a named place into a raster window.
    """
    arrays, metadata = _load_cascast_ensemble(event_id)
    reference = _catalog_vil_event(event_id)
    box = region or [0, 0, int(arrays["ground_truth"].shape[1]), int(arrays["ground_truth"].shape[2])]
    if len(box) != 4:
        raise ValueError("region must be [row0, col0, row1, col1]")
    row0, col0, row1, col1 = [int(value) for value in box]
    height, width = arrays["ground_truth"].shape[1:]
    row0, col0 = max(0, row0), max(0, col0)
    row1, col1 = min(height, row1), min(width, col1)
    if row0 >= row1 or col0 >= col1:
        raise ValueError("region does not intersect the 384x384 event grid")

    forecast = arrays["ensemble_mean"][:12, row0:row1, col0:col1]
    truth = arrays["ground_truth"][:12, row0:row1, col0:col1]
    forecast_metrics = _comparison_metrics(forecast, threshold)
    truth_metrics = _comparison_metrics(truth, threshold)
    try:
        geographic_region = geospatial.pixel_box_to_latlon_bounds(
            event_id, [row0, col0, row1, col1]
        )
        geographic_alignment = "per_event_geospatial"
    except (OSError, RuntimeError, ValueError, KeyError):
        # Lightweight/mocked backends and legacy records may not expose
        # projection metadata. Preserve the primitive's old pixel-box behavior
        # while making the limitation explicit in the returned provenance.
        geographic_region = None
        geographic_alignment = "reference_pixel_box_fallback"

    reference_month = reference.get("month")
    reference_time = datetime.fromisoformat(str(reference["time_utc"]).replace("Z", "+00:00"))
    month_window = max(0, min(int(month_window), 6))
    normalized_day_window = None if day_window is None else max(0, min(int(day_window), 366))
    candidates: list[dict[str, Any]] = []
    for candidate in _catalog_vil_records():
        if candidate["id"] == reference["id"]:
            continue
        candidate_time = datetime.fromisoformat(str(candidate["time_utc"]).replace("Z", "+00:00"))
        if candidate_time.year == reference_time.year:
            continue
        if normalized_day_window is not None:
            reference_anchor = reference_time.replace(year=2000)
            candidate_anchor = candidate_time.replace(year=2000)
            day_distance = abs((candidate_anchor - reference_anchor).days)
            day_distance = min(day_distance, 366 - day_distance)
            if day_distance > normalized_day_window:
                continue
        elif reference_month is not None and candidate["month"] is not None:
            distance = abs(int(candidate["month"]) - int(reference_month))
            distance = min(distance, 12 - distance)
            if distance > month_window:
                continue
        if any(candidate.get(key) is None for key in ("llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon")):
            continue
        overlaps = not (
            float(candidate["urcrnrlat"]) < float(reference["llcrnrlat"])
            or float(candidate["llcrnrlat"]) > float(reference["urcrnrlat"])
            or float(candidate["urcrnrlon"]) < float(reference["llcrnrlon"])
            or float(candidate["llcrnrlon"]) > float(reference["urcrnrlon"])
        )
        if overlaps:
            candidates.append(candidate)
    candidates.sort(key=lambda item: abs(
        (datetime.fromisoformat(str(item["time_utc"]).replace("Z", "+00:00")).replace(year=2000)
         - reference_time.replace(year=2000)).days
    ))
    candidates = candidates[: max(1, min(int(candidate_limit), 20))]

    historical_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            observed, source = _load_vil_array(candidate)
            if geographic_region is None:
                candidate_region = {"status": "ok", "pixel_box": [row0, col0, row1, col1]}
            else:
                candidate_region = geospatial.latlon_box_to_pixel_box(
                    candidate["id"], geographic_region
                )
            if candidate_region.get("status") != "ok":
                historical_rows.append({
                    "event_id": candidate["id"],
                    "available": False,
                    "reason": "geographic region does not intersect candidate grid",
                    "region_pixel_box": candidate_region.get("pixel_box"),
                })
                continue
            candidate_box = candidate_region["pixel_box"]
            candidate_row0, candidate_col0, candidate_row1, candidate_col1 = candidate_box
            observed = observed[
                : max(1, min(int(history_frame_count), observed.shape[0])),
                candidate_row0:candidate_row1,
                candidate_col0:candidate_col1,
            ]
            metrics = _comparison_metrics(observed, threshold)
        except (OSError, RuntimeError, ValueError, IndexError, KeyError) as error:
            historical_rows.append({"event_id": candidate["id"], "available": False, "reason": str(error)})
            continue
        historical_rows.append({
            "event_id": candidate["id"],
            "available": True,
            "time_utc": candidate["time_utc"],
            "month": candidate["month"],
            "event_type": candidate["event_type"],
            "source_hdf5": source,
            "region_pixel_box": candidate_region["pixel_box"],
            "metrics": metrics,
        })

    available = [row["metrics"] for row in historical_rows if row.get("available")]
    historical_summary: dict[str, Any] = {
        "sample_count": len(available),
        "candidate_count": len(candidates),
        "coverage_years": sorted({
            int(row["time_utc"][:4]) for row in historical_rows
            if row.get("available") and row.get("time_utc")
        }),
    }
    for metric_name in ("max_vil", "mean_vil", "p95_vil", "threshold_coverage"):
        values = [item[metric_name]["max"] for item in available if item[metric_name]["max"] is not None]
        if values:
            current = forecast_metrics[metric_name]["max"]
            historical_summary[metric_name] = {
                "p10": round(float(np.quantile(values, 0.10)), 4),
                "median": round(float(np.quantile(values, 0.50)), 4),
                "p90": round(float(np.quantile(values, 0.90)), 4),
                "current_forecast": current,
                "current_percentile": round(float(np.mean(np.asarray(values) <= current) * 100), 2) if current is not None else None,
            }
        else:
            historical_summary[metric_name] = {"p10": None, "median": None, "p90": None, "current_forecast": forecast_metrics[metric_name]["max"], "current_percentile": None}

    return {
        "status": "ok",
        "task": "cascast_historical_region_comparison",
        "event_id": event_id,
        "region_pixel_box": [row0, col0, row1, col1],
        "region_latlon_bounds": geographic_region,
        "reference_time_utc": reference["time_utc"],
        "reference_month": reference_month,
        "month_window": month_window,
        "day_window": normalized_day_window,
        "other_years_only": True,
        "threshold": float(threshold),
        "forecast": {"model": "cascast_ensemble_mean", "metrics": forecast_metrics},
        "ground_truth": {"available": True, "metrics": truth_metrics},
        "historical": historical_summary,
        "historical_events": historical_rows,
        "provenance": {
            "forecast": "cached CasCast ensemble mean",
            "ground_truth": "aligned held-out SEVIR observations",
            "history": "CATALOG spatially overlapping events from other years within the requested day window and on-demand HDF5 reads",
            "historical_region_alignment": geographic_alignment,
            "missing_value": 255,
        },
        "interpretation_limit": (
            "历史样本是SEVIR中可检索且与参考事件地理范围相交的有限样本，不是完整气候统计；"
            "百分位是经验样本百分位，不是概率预报。"
        ),
    }


def load_event_history_frame_primitives(
    event_ids: list[str],
    start_frame: int = 0,
    end_frame: int = 49,
) -> dict[str, Any]:
    """Load per-frame primitive observations for Agent-selected history analyses.

    This is deliberately not a duration report.  For each requested event it
    exposes the original five-minute timeline together with per-frame valid
    pixel maxima and counts.  Generated Agent code chooses the threshold,
    activity rule, run-length calculation, and group statistic.  Missing local
    HDF5 rows are reported per event instead of being silently dropped.
    """
    requested = [str(value).strip() for value in (event_ids or []) if str(value).strip()]
    if not requested:
        raise ValueError("event_ids must contain at least one SEVIR event")
    if len(requested) > 20:
        raise ValueError("event_ids is limited to 20 events")
    start = max(0, int(start_frame))
    end = min(49, int(end_frame))
    if start >= end:
        raise ValueError("start_frame must be smaller than end_frame")

    records: list[dict[str, Any]] = []
    for event_id in requested:
        record = _catalog_vil_event(event_id)
        try:
            frames, source_path = _load_vil_array(record)
        except (FileNotFoundError, IndexError, RuntimeError, ValueError) as error:
            records.append({
                "event_id": record["id"],
                "available": False,
                "error": str(error),
                "file_name": record["file_name"],
                "file_index": record["file_index"],
            })
            continue
        selected = frames[start:end]
        offsets = list(record["minute_offsets"][start:end])
        frame_primitives: list[dict[str, Any]] = []
        for index, frame in enumerate(selected, start=start):
            valid = frame[frame != 255]
            frame_primitives.append({
                "frame_index": int(index),
                "relative_minutes": int(record["minute_offsets"][index]),
                "max_vil": int(valid.max()) if valid.size else None,
                "valid_pixel_count": int(valid.size),
                "missing_pixel_count": int((frame == 255).sum()),
            })
        records.append({
            "event_id": record["id"],
            "available": True,
            "event_type": record["event_type"],
            "data_max": record["data_max"],
            "file_name": record["file_name"],
            "file_index": record["file_index"],
            "source_hdf5": str(source_path),
            "frame_interval_minutes": 5,
            "window_minutes": [offsets[0], offsets[-1]],
            "frames": frame_primitives,
            "primitive_note": (
                "Per-frame maxima and validity counts only; duration and all group "
                "statistics must be calculated by the Agent."
            ),
        })
    return {
        "status": "ok",
        "task": "event_history_frame_primitives",
        "capabilities": ["observed_timeline", "frame_validity", "frame_maximum"],
        "provenance": _catalog_provenance(source_kind="catalog_indexed_local_sevir_hdf5"),
        "requested_event_ids": requested,
        "frame_interval_minutes": 5,
        "frame_range": [start, end],
        "records": records,
        "statistics_computed": [],
        "interpretation_limit": (
            "The tool exposes per-frame source-derived primitives, not a fixed "
            "duration or similarity answer."
        ),
    }


def load_event_history_raw(
    event_ids: list[str],
    start_frame: int = 0,
    end_frame: int = 49,
    spatial_stride: int = 1,
    max_total_values: int = 8_000_000,
) -> dict[str, Any]:
    """Return bounded source frames for Agent-defined historical calculations.

    No threshold, object definition, similarity score, duration, or aggregate is
    chosen here.  The caller may derive those quantities from ``frame_values``.
    A strict value budget prevents a generated program from accidentally loading
    many full 384x384 event cubes in one call; it may make several smaller calls.
    ``end_frame`` is exclusive, matching normal Python slicing.
    """
    requested = [str(value).strip() for value in (event_ids or []) if str(value).strip()]
    if not requested:
        raise ValueError("event_ids must contain at least one SEVIR event")
    if len(requested) > 20:
        raise ValueError("event_ids is limited to 20 events")
    start = max(0, int(start_frame))
    end = min(49, int(end_frame))
    if start >= end:
        raise ValueError("start_frame must be smaller than end_frame")
    stride = int(spatial_stride)
    if stride not in {1, 2, 4, 8}:
        raise ValueError("spatial_stride must be one of 1, 2, 4, or 8")
    value_budget = max(1, min(int(max_total_values), 8_000_000))

    records: list[dict[str, Any]] = []
    loaded_cubes: list[np.ndarray] = []
    loaded_event_ids: list[str] = []
    loaded_timelines: list[list[int]] = []
    loaded_values = 0
    for event_id in requested:
        record = _catalog_vil_event(event_id)
        try:
            frames, source_path = _load_vil_array(record)
        except (FileNotFoundError, IndexError, RuntimeError, ValueError) as error:
            records.append({
                "event_id": record["id"],
                "available": False,
                "error": str(error),
                "file_name": record["file_name"],
                "file_index": record["file_index"],
            })
            continue

        selected = np.ascontiguousarray(
            frames[start:end, ::stride, ::stride], dtype=np.uint8
        )
        if loaded_values + int(selected.size) > value_budget:
            records.append({
                "event_id": record["id"],
                "available": False,
                "error": "raw frame value budget would be exceeded; use a larger spatial_stride, a shorter frame window, or a separate call",
                "required_values": int(selected.size),
                "remaining_value_budget": max(0, value_budget - loaded_values),
            })
            continue
        loaded_values += int(selected.size)
        selected.setflags(write=False)
        offsets = [int(value) for value in record["minute_offsets"][start:end]]
        missing_counts = (selected == 255).sum(axis=(1, 2)).astype(np.int64).tolist()
        loaded_cubes.append(selected)
        loaded_event_ids.append(record["id"])
        loaded_timelines.append(offsets)
        records.append({
            "event_id": record["id"],
            "available": True,
            "event_type": record.get("event_type"),
            "file_name": record.get("file_name"),
            "file_index": record.get("file_index"),
            "source_hdf5": str(source_path),
            "frame_indices": list(range(start, end)),
            "relative_minutes": offsets,
            "frame_interval_minutes": (
                int(offsets[1] - offsets[0]) if len(offsets) > 1 else None
            ),
            "frame_shape": [int(selected.shape[1]), int(selected.shape[2])],
            "frame_count": int(selected.shape[0]),
            "dtype": "uint8",
            "missing_value_code": 255,
            "missing_pixel_count": [int(value) for value in missing_counts],
            "spatial_stride": stride,
            "frame_values": selected,
            "array_transport": "read_only_ndarray_inside_isolated_worker",
            "observation_window": {
                "first_relative_minute": offsets[0] if offsets else None,
                "last_relative_minute": offsets[-1] if offsets else None,
                "left_boundary_observed": True,
                "right_boundary_observed": True,
                "censoring_rule": (
                    "For any Agent-defined activity mask, a run touching the first "
                    "selected frame is left-censored and a run touching the last "
                    "selected frame is right-censored. Such a duration is a lower bound."
                ),
                "complete_storm_lifecycle_available": False,
            },
        })
    cube_shapes = {tuple(value.shape) for value in loaded_cubes}
    frame_cube_available = bool(loaded_cubes) and len(cube_shapes) == 1
    frame_cube = None
    if frame_cube_available:
        frame_cube = np.ascontiguousarray(np.stack(loaded_cubes, axis=0), dtype=np.uint8)
        frame_cube.setflags(write=False)
    shared_timeline = (
        loaded_timelines[0]
        if loaded_timelines and all(value == loaded_timelines[0] for value in loaded_timelines)
        else None
    )
    return {
        "status": "ok",
        "task": "event_history_raw_frames",
        "capabilities": [
            "raw_observed_frames",
            "agent_defined_threshold",
            "agent_defined_features",
            "agent_defined_similarity",
            "agent_defined_duration",
        ],
        "provenance": _catalog_provenance(source_kind="catalog_indexed_local_sevir_hdf5"),
        "requested_event_ids": requested,
        "frame_range": [start, end],
        "frame_range_semantics": "start inclusive, end exclusive",
        "spatial_stride": stride,
        "max_total_values": value_budget,
        "loaded_value_count": loaded_values,
        "loaded_event_count": len(loaded_event_ids),
        "loaded_event_ids": loaded_event_ids,
        "frame_cube_available": frame_cube_available,
        "frame_cube": frame_cube,
        "frame_cube_layout": "[event,time,row,column] aligned with loaded_event_ids",
        "frame_cube_shape": (
            [int(value) for value in frame_cube.shape] if frame_cube is not None else None
        ),
        "relative_minutes": shared_timeline,
        "missing_value_code": 255,
        "array_usage_rule": (
            "frame_cube and records[*].frame_values are ndarrays. Never use `if array`, "
            "`if not array`, or compare an array directly as a scalar. Check "
            "frame_cube_available/frame_count and use arrays.any/all/count_nonzero."
        ),
        "records": records,
        "statistics_computed": [],
        "threshold_applied": None,
        "similarity_definition_applied": None,
        "duration_definition_applied": None,
        "interpretation_limit": (
            "These are bounded encoded-VIL observations. The tool does not establish "
            "meteorological similarity or a complete storm lifecycle."
        ),
    }


def aggregate_sevir_records(
    records: dict[str, Any] | list[dict[str, Any]],
    field: str,
    operations: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """Compute agent-selected aggregates over a query result."""
    if field not in _runtime_numeric_fields():
        raise ValueError(f"unsupported aggregation field: {field}")
    rows = records.get("records", []) if isinstance(records, dict) else records
    if not isinstance(rows, list):
        raise ValueError("records must be a query result or list of dictionaries")
    values = [row.get(field) for row in rows if isinstance(row, dict) and row.get(field) is not None]
    numeric = np.asarray(values, dtype=np.float64) if values else np.asarray([], dtype=np.float64)
    requested = list(operations or ["count", "mean", "median"])
    unknown = sorted(set(requested) - AGGREGATE_OPERATIONS)
    if unknown:
        raise ValueError(f"unsupported aggregate operations: {', '.join(unknown)}")
    result: dict[str, Any] = {
        "status": "ok",
        "task": "catalog_aggregate",
        "capabilities": ["agent_selected_aggregation"],
        "provenance": _catalog_provenance(),
        "field": field,
        "valid_value_count": int(numeric.size),
    }
    for operation in requested:
        if operation == "count":
            result[operation] = int(numeric.size)
        elif not numeric.size:
            result[operation] = None
        elif operation == "mean":
            result[operation] = round(float(numeric.mean()), 4)
        elif operation == "min":
            result[operation] = round(float(numeric.min()), 4)
        elif operation == "max":
            result[operation] = round(float(numeric.max()), 4)
        elif operation == "median":
            result[operation] = round(float(np.median(numeric)), 4)
        elif operation == "std":
            result[operation] = round(float(numeric.std()), 4)
        elif operation == "sum":
            result[operation] = round(float(numeric.sum()), 4)
        elif operation == "quantile":
            qs = [float(q) for q in (quantiles or [0.5])]
            if any(q < 0 or q > 1 for q in qs):
                raise ValueError("quantiles must be between 0 and 1")
            result["quantiles"] = {
                str(q): round(float(np.quantile(numeric, q)), 4) for q in qs
            }
    return result


def rank_sevir_records(
    records: dict[str, Any] | list[dict[str, Any]],
    field: str,
    limit: int = 10,
    descending: bool = True,
) -> dict[str, Any]:
    """Rank a query result by a selected field with a bounded output."""
    if field not in _runtime_numeric_fields():
        raise ValueError(f"unsupported ranking field: {field}")
    rows = records.get("records", []) if isinstance(records, dict) else records
    ranked = [row for row in rows if isinstance(row, dict) and row.get(field) is not None]
    ranked.sort(key=lambda row: row[field], reverse=bool(descending))
    bounded_limit = max(1, min(int(limit), 100))
    return {
        "status": "ok",
        "task": "catalog_rank",
        "capabilities": ["agent_selected_ranking"],
        "provenance": _catalog_provenance(),
        "field": field,
        "descending": bool(descending),
        "limit": bounded_limit,
        "records": ranked[:bounded_limit],
    }


def aggregate_sevir_catalog(
    year: int | None = None,
    event_type: str = "",
    field: str = "data_max",
    operations: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate the complete filtered catalog without returning all rows."""
    query = query_sevir_records(
        year=year,
        event_type=event_type,
        fields=[field],
        limit=50000,
    )
    result = aggregate_sevir_records(query, field, operations, quantiles)
    result["filters"] = {"year": year, "event_type": event_type}
    result["source_record_count"] = query["record_count"]
    result["truncated"] = query["truncated"]
    return result


def rank_sevir_catalog(
    year: int | None = None,
    event_type: str = "",
    field: str = "data_max",
    limit: int = 10,
    descending: bool = True,
) -> dict[str, Any]:
    """Rank the complete filtered catalog while returning only the requested rows."""
    query = query_sevir_records(
        year=year,
        event_type=event_type,
        fields=None,
        limit=50000,
    )
    result = rank_sevir_records(query, field, limit, descending)
    result["filters"] = {"year": year, "event_type": event_type}
    result["source_record_count"] = query["record_count"]
    return result


def group_sevir_catalog(
    group_by: str,
    field: str = "data_max",
    year: int | None = None,
    event_type: str = "",
    operations: list[str] | None = None,
    quantiles: list[float] | None = None,
    group_limit: int = 50,
) -> dict[str, Any]:
    """Group filtered catalog rows and apply only agent-selected aggregates."""
    if group_by not in _runtime_catalog_fields():
        raise ValueError(f"unsupported grouping field: {group_by}")
    if field not in _runtime_numeric_fields():
        raise ValueError(f"unsupported aggregation field: {field}")
    normalized_type = str(event_type or "").strip().lower()
    rows = [
        record
        for record in _catalog_vil_records()
        if (year is None or record["year"] == int(year))
        and (not normalized_type or normalized_type in str(record["event_type"]).lower())
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_by) if row.get(group_by) is not None else "Unknown")
        grouped.setdefault(key, []).append(row)
    bounded_limit = max(1, min(int(group_limit), 200))
    ordered_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    summaries = []
    for key, group_rows in ordered_groups[:bounded_limit]:
        summary = aggregate_sevir_records(group_rows, field, operations, quantiles)
        summaries.append(
            {
                "group": key,
                "record_count": len(group_rows),
                "statistics": summary,
            }
        )
    return {
        "status": "ok",
        "task": "catalog_group_aggregate",
        "capabilities": ["agent_selected_grouping", "agent_selected_aggregation"],
        "provenance": _catalog_provenance(),
        "group_by": group_by,
        "field": field,
        "filters": {"year": year, "event_type": event_type},
        "source_record_count": len(rows),
        "group_count": len(grouped),
        "returned_group_count": len(summaries),
        "truncated": len(grouped) > bounded_limit,
        "groups": summaries,
        "interpretation_limit": (
            "Groups and values come from SEVIR catalog metadata; record counts are not "
            "unique disaster counts."
        ),
    }


def compare_sevir_catalog_groups(
    group_by: str,
    group_values: list[str],
    field: str = "data_max",
    year: int | None = None,
    operations: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """Compare explicitly selected catalog groups without choosing metrics for the agent."""
    if group_by not in _runtime_catalog_fields():
        raise ValueError(f"unsupported grouping field: {group_by}")
    requested = [str(value) for value in group_values]
    if not requested:
        raise ValueError("group_values must contain at least one value")
    if len(requested) > 20:
        raise ValueError("group_values is limited to 20 entries")
    rows = [
        row
        for row in _catalog_vil_records()
        if (year is None or row["year"] == int(year))
    ]
    summaries = []
    for requested_value in requested:
        group_rows = [row for row in rows if str(row.get(group_by)) == requested_value]
        summary = aggregate_sevir_records(group_rows, field, operations, quantiles)
        summaries.append(
            {
                "group": requested_value,
                "record_count": len(group_rows),
                "statistics": summary,
            }
        )
    return {
        "status": "ok",
        "task": "catalog_group_comparison",
        "capabilities": ["agent_selected_group_comparison"],
        "provenance": _catalog_provenance(),
        "group_by": group_by,
        "field": field,
        "year": year,
        "groups": summaries,
        "interpretation_limit": (
            "This comparison uses catalog metadata and the exact groups requested by the agent."
        ),
    }


def historical_sevir_statistics(
    year: int | None = None,
    event_type: str = "",
) -> dict[str, Any]:
    event_type_normalized = str(event_type or "").strip().lower()
    records = [
        record
        for record in _catalog_vil_records()
        if (year is None or record["year"] == int(year))
        and (not event_type_normalized or event_type_normalized in record["event_type"].lower())
    ]
    if not records:
        raise ValueError("当前筛选条件下没有SEVIR VIL catalog记录。")
    counts_by_year: dict[str, int] = {}
    counts_by_month: dict[str, int] = {}
    counts_by_event_type: dict[str, int] = {}
    for record in records:
        year_key = str(record["year"] or "Unknown")
        month_key = f"{record['month']:02d}" if record["month"] else "Unknown"
        type_key = record["event_type"]
        counts_by_year[year_key] = counts_by_year.get(year_key, 0) + 1
        counts_by_month[month_key] = counts_by_month.get(month_key, 0) + 1
        counts_by_event_type[type_key] = counts_by_event_type.get(type_key, 0) + 1
    max_values = np.asarray(
        [record["data_max"] for record in records if record["data_max"] is not None],
        dtype=np.float64,
    )
    max_summary = None
    if max_values.size:
        max_summary = {
            "count": int(max_values.size),
            "mean": round(float(max_values.mean()), 4),
            "min": round(float(max_values.min()), 4),
            "p50": round(float(np.percentile(max_values, 50)), 4),
            "p90": round(float(np.percentile(max_values, 90)), 4),
            "p95": round(float(np.percentile(max_values, 95)), 4),
            "max": round(float(max_values.max()), 4),
        }
    ranked = sorted(
        (record for record in records if record["data_max"] is not None),
        key=lambda item: item["data_max"],
        reverse=True,
    )[:10]
    return {
        "status": "ok",
        "task": "historical_sevir_statistics",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "filters": {"year": year, "event_type": event_type},
        "record_count": len(records),
        "counts_by_year": dict(sorted(counts_by_year.items())),
        "counts_by_month": dict(sorted(counts_by_month.items())),
        "counts_by_event_type": dict(
            sorted(counts_by_event_type.items(), key=lambda item: item[1], reverse=True)
        ),
        "data_max_summary": max_summary,
        "top_events_by_catalog_data_max": [_catalog_public_record(record) for record in ranked],
        "interpretation_limit": (
            "这是CATALOG记录级统计。data_max是每条VIL记录的catalog最大值；"
            "事件数量不等同于灾害发生次数，Unlabeled也不代表无灾害。"
        ),
    }


def get_noaa_storm_reference(event_id: str) -> dict[str, Any]:
    record = _catalog_vil_event(event_id)
    noaa_event_id = record["noaa_event_id"]
    noaa_episode_id = record["noaa_episode_id"]
    linked = noaa_event_id is not None
    return {
        "status": "ok",
        "task": "noaa_storm_reference",
        "sevir_id": record["id"],
        "event_type": record["event_type"],
        "noaa_linked": linked,
        "noaa_episode_id": noaa_episode_id,
        "noaa_event_id": noaa_event_id,
        "noaa_event_url": (
            f"https://www.ncei.noaa.gov/stormevents/eventdetails.jsp?id={noaa_event_id}"
            if linked
            else None
        ),
        "reason": (
            "CATALOG contains NOAA Storm Events identifiers for this storm event."
            if linked
            else "This record has no NOAA identifiers; random SEVIR events are commonly unlabeled."
        ),
        "interpretation_limit": (
            "This tool exposes the CATALOG linkage and official NOAA detail URL. It does not copy "
            "or independently verify the NOAA narrative, damage, or warning fields."
        ),
    }


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool).tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def compare_event_with_history(
    event_id: str,
    threshold: float = 133.0,
) -> dict[str, Any]:
    target = str(event_id).strip().lower()
    records = _catalog_vil_records()
    record = next((item for item in records if item["id"].lower() == target), None)
    if record is None:
        raise ValueError(f"SEVIR CATALOG中没有找到VIL事件{event_id}。")
    if record["data_max"] is None:
        raise ValueError(f"事件{event_id}没有可比较的catalog data_max。")
    reference = [
        item["data_max"]
        for item in records
        if item["data_max"] is not None
        and (record["event_type"] == "Unlabeled" or item["event_type"] == record["event_type"])
    ]
    values = np.asarray(reference, dtype=np.float64)
    percentile = float((values <= record["data_max"]).mean() * 100)
    observed_history: dict[str, Any] = {
        "available": False,
        "reason": "当前事件没有已接入的逐帧历史观测，因此catalog metadata不能给出持续时间。",
    }
    try:
        history, source_path = _load_vil_array(record)
        minute_offsets = list(record["minute_offsets"])
        observation_source = "official_sevir_hdf5"
        observation_source_detail = str(source_path)
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as hdf5_error:
        try:
            arrays, metadata = _load_cascast_ensemble(record["id"])
            history = arrays["history"].astype(np.float32)
            minute_offsets = list(range(-(history.shape[0] - 1) * 5, 1, 5))
            observation_source = "saved_event_history_observations"
            observation_source_detail = str(metadata.get("cache_file", ""))
        except (KeyError, RuntimeError, ValueError):
            observed_history = {
                "available": False,
                "reason": (
                    "Catalog metadata was found, but neither its indexed local HDF5 row "
                    "nor a saved observation history could be loaded."
                ),
                "hdf5_error": str(hdf5_error),
            }
    else:
        pass

    if observed_history.get("hdf5_error") is None:
        valid = history != 255
        active = ((history >= threshold) & valid).any(axis=(1, 2))
        longest_frames = _longest_true_run(active)
        active_indices = [int(index) for index, flag in enumerate(active.tolist()) if flag]
        frame_interval_minutes = (
            int(minute_offsets[1] - minute_offsets[0]) if len(minute_offsets) > 1 else 5
        )
        window_duration_minutes = max(
            0, int(minute_offsets[-1] - minute_offsets[0]) if minute_offsets else 0
        )
        first_active_index = active_indices[0] if active_indices else None
        last_active_index = active_indices[-1] if active_indices else None
        first_active_minute = (
            int(minute_offsets[first_active_index]) if first_active_index is not None else None
        )
        last_active_minute = (
            int(minute_offsets[last_active_index]) if last_active_index is not None else None
        )
        observed_history = {
            "available": True,
            "source": observation_source,
            "source_detail": observation_source_detail,
            "file_name": record["file_name"],
            "file_index": record["file_index"],
            "window_minutes": [minute_offsets[0], minute_offsets[-1]],
            "frame_interval_minutes": frame_interval_minutes,
            "frame_count": int(history.shape[0]),
            "threshold": float(threshold),
            "threshold_exceedance_rule": "any_valid_pixel_at_or_above_threshold",
            "active_frame_count": int(active.sum()),
            "first_active_minute": first_active_minute,
            "last_active_minute": last_active_minute,
            "active_time_span_minutes": (
                int(last_active_minute - first_active_minute)
                if first_active_minute is not None and last_active_minute is not None
                else 0
            ),
            # Retained for compatibility. This is sampled active support, not a
            # claim about a continuous or complete meteorological lifecycle.
            "total_active_minutes": min(
                int(active.sum() * frame_interval_minutes), window_duration_minutes
            ),
            "longest_consecutive_minutes": (
                max(0, int((longest_frames - 1) * frame_interval_minutes))
                if longest_frames else 0
            ),
            "touches_window_start": bool(active[0]) if active.size else False,
            "touches_window_end": bool(active[-1]) if active.size else False,
            "left_censored": bool(active[0]) if active.size else False,
            "right_censored": bool(active[-1]) if active.size else False,
            "duration_is_lower_bound": bool(active[0] or active[-1]) if active.size else False,
            "threshold_episode_captured_within_window": bool(
                active.size and active.any() and not active[0] and not active[-1]
            ),
            "complete_storm_lifecycle_available": False,
            "duration_estimator": (
                "elapsed time between the first and last samples in the longest "
                "consecutive threshold-active run"
            ),
            "interpretation_limit": (
                "持续时间来自catalog定位到的真实逐帧VIL观测，不是CasCast未来预测。"
                "每帧只判断是否存在有效像素达到阈值，不使用隐藏的最小像素数条件。"
            ),
        }
    return {
        "status": "ok",
        "task": "event_history_comparison",
        "capabilities": ["catalog_percentile", "threshold_episode_observation"],
        "provenance": _catalog_provenance(source_kind="catalog_and_indexed_observation"),
        "event_id": record["id"],
        "event_type": record["event_type"],
        "time_utc": record["time_utc"],
        "catalog_data_max": record["data_max"],
        "reference_group": record["event_type"] if record["event_type"] != "Unlabeled" else "all_vil_records",
        "reference_count": int(values.size),
        "data_max_percentile": round(percentile, 2),
        "observed_high_vil_duration": observed_history,
        "interpretation": f"该记录的catalog data_max高于或等于约{percentile:.2f}%的参考记录。",
        "interpretation_limit": (
            "分位数来自catalog data_max的经验分布，不是灾害概率或模型预测概率；"
            "逐帧持续时间来自catalog所索引的本地HDF5，而不是catalog字段本身。"
        ),
    }


def compare_observed_event_with_historical_window(
    event_id: str,
    day_window: int = 15,
    threshold: float = 133.0,
    candidate_limit: int | None = None,
    display_limit: int = 20,
) -> dict[str, Any]:
    """Compare any indexed SEVIR event with same-region, other-year events.

    This path uses observed SEVIR HDF5 data and does not require a CasCast cache.
    The reference footprint is converted to WGS84 and reprojected for every
    candidate before metrics are calculated.
    """
    reference = _catalog_vil_event(str(event_id))
    reference_values, reference_source = _load_vil_array(reference)
    reference_box = [0, 0, int(reference_values.shape[1]), int(reference_values.shape[2])]
    region_bounds = geospatial.pixel_box_to_latlon_bounds(reference["id"], reference_box)
    reference_time = datetime.fromisoformat(str(reference["time_utc"]).replace("Z", "+00:00"))
    window = max(0, min(int(day_window), 366))

    def spatial_structure(values: np.ndarray) -> dict[str, Any]:
        clean = np.where(values == 255, -np.inf, values.astype(np.float32))
        frame_index = int(np.argmax(np.max(clean, axis=(1, 2))))
        frame = values[frame_index]
        mask = (frame != 255) & (frame >= float(threshold))
        rows, columns = np.where(mask)
        result: dict[str, Any] = {
            "peak_frame_index": frame_index,
            "high_value_area_pixels": int(mask.sum()),
            "centroid_row": None,
            "centroid_column": None,
            "bounding_box_pixels": None,
        }
        if rows.size:
            result["centroid_row"] = round(float(rows.mean()), 3)
            result["centroid_column"] = round(float(columns.mean()), 3)
            result["bounding_box_pixels"] = [
                int(rows.min()), int(columns.min()), int(rows.max() + 1), int(columns.max() + 1)
            ]
        return result

    reference_metrics = _comparison_metrics(reference_values, threshold)
    reference_structure = spatial_structure(reference_values)
    candidates: list[dict[str, Any]] = []
    for candidate in _catalog_vil_records():
        if candidate["id"] == reference["id"] or not candidate.get("time_utc"):
            continue
        candidate_time = datetime.fromisoformat(str(candidate["time_utc"]).replace("Z", "+00:00"))
        if candidate_time.year == reference_time.year:
            continue
        reference_anchor = reference_time.replace(year=2000)
        candidate_anchor = candidate_time.replace(year=2000)
        distance = abs((candidate_anchor - reference_anchor).days)
        if min(distance, 366 - distance) > window:
            continue
        bounds = [candidate.get(key) for key in ("llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon")]
        if any(value is None for value in bounds):
            continue
        if not (
            float(candidate["urcrnrlat"]) >= float(region_bounds["south"])
            and float(candidate["llcrnrlat"]) <= float(region_bounds["north"])
            and float(candidate["urcrnrlon"]) >= float(region_bounds["west"])
            and float(candidate["llcrnrlon"]) <= float(region_bounds["east"])
        ):
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: abs(
        (datetime.fromisoformat(str(item["time_utc"]).replace("Z", "+00:00")).replace(year=2000)
         - reference_time.replace(year=2000)).days
    ))
    if candidate_limit is not None:
        candidates = candidates[: max(1, min(int(candidate_limit), 50000))]

    historical_events: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_years: list[int] = []
    for candidate in candidates:
        try:
            values, source = _load_vil_array(candidate)
            candidate_region = geospatial.latlon_box_to_pixel_box(candidate["id"], region_bounds)
            if candidate_region.get("status") != "ok":
                continue
            row0, col0, row1, col1 = candidate_region["pixel_box"]
            values = values[:, row0:row1, col0:col1]
            detail = {
                "event_id": candidate["id"],
                "time_utc": candidate["time_utc"],
                "year": int(str(candidate["time_utc"])[:4]),
                "source_hdf5": source,
                "metrics": _comparison_metrics(values, threshold),
                "spatial_structure": spatial_structure(values),
            }
            all_metrics.append(detail["metrics"])
            all_years.append(detail["year"])
            if len(historical_events) < max(0, int(display_limit)):
                historical_events.append(detail)
        except (OSError, RuntimeError, ValueError, IndexError, KeyError):
            continue

    available = [item for item in all_metrics if item]
    summary: dict[str, Any] = {
        "candidate_count": len(candidates),
        "sample_count": len(available),
        "coverage_years": sorted(set(all_years)),
    }
    for name in ("max_vil", "p95_vil", "threshold_coverage"):
        values = [item[name]["max"] for item in available if item.get(name, {}).get("max") is not None]
        summary[name] = {
            "median": round(float(np.median(values)), 4) if values else None,
            "p10": round(float(np.quantile(values, 0.10)), 4) if values else None,
            "p90": round(float(np.quantile(values, 0.90)), 4) if values else None,
        }
    return {
        "status": "ok",
        "task": "observed_event_historical_window_comparison",
        "event_id": reference["id"],
        "reference": {
            "time_utc": reference["time_utc"],
            "event_type": reference["event_type"],
            "source_hdf5": reference_source,
            "region_latlon_bounds": region_bounds,
            "metrics": reference_metrics,
            "spatial_structure": reference_structure,
        },
        "historical": summary,
        "historical_events": historical_events,
        "filters": {
            "day_window": window,
            "other_years_only": True,
            "spatial_overlap": True,
            "candidate_limit": candidate_limit,
            "display_limit": max(0, int(display_limit)),
        },
        "display": {
            "displayed_count": len(historical_events),
            "details_truncated": len(available) > len(historical_events),
        },
        "provenance": {"catalog": "SEVIR CATALOG", "observations": "remote_or_local_SEVIR_HDF5", "region_alignment": "WGS84_bounds_reprojected_per_event"},
        "interpretation_limit": "历史样本是SEVIR中可检索的有限样本，不是完整气候统计；百分位是经验分布，不是概率。",
    }
