"""Small, single-purpose adapters at the public capability boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..capabilities.models import Event, UploadMatch
from ..capabilities.dataframe_schema import (
    DataFrameSchemaError,
    GEO_PLACE_CANDIDATE,
    normalize_dataframe,
)


def optional_number(value: Any) -> float | None:
    """Convert an optional scalar to ``float`` while preserving missing values.

    Parameters
    ----------
    value : Any
        Scalar or missing value.
    Returns
    -------
    float or None
        Normalized scalar.
    """

    return None if value in (None, "") else float(value)


def utc_text(value: Any) -> str:
    """Normalize a timestamp to ISO-8601 UTC text.

    Parameters
    ----------
    value : Any
        Timestamp-like input.
    Returns
    -------
    str
        UTC timestamp or an empty string.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def require_utc_text(value: Any, field_name: str) -> str:
    """Validate an explicit timezone-aware ISO timestamp and normalize to UTC."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    timezone_hint = (
        "use Z for UTC (e.g. '2017-09-12T00:00:00Z') or an explicit UTC offset; "
        "resolve the local timezone before converting local times"
    )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be an ISO-8601 timestamp with an explicit timezone; "
            f"{timezone_hint}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an explicit timezone; {timezone_hint}")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _edges(value: Any) -> tuple[float | None, float | None, float | None, float | None]:
    """Read south, west, north and east from a raw mapping.

    Parameters
    ----------
    value : Any
        Raw bounding-box mapping.
    Returns
    -------
    tuple
        Four normalized edges.
    """
    if not isinstance(value, dict):
        return (None, None, None, None)
    return tuple(optional_number(value.get(key)) for key in ("south", "west", "north", "east"))  # type: ignore[return-value]


def catalog_row(record: dict[str, Any]) -> dict[str, Any]:
    """Prepare one source-native catalog record for schema normalization.

    Parameters
    ----------
    record : dict
        Raw catalog record.
    Returns
    -------
    dict
        Source fields plus derived availability and flattened bounding-box
        aliases. Canonical/legacy pairs remain separate so the DataFrame
        schema can detect conflicts instead of silently choosing one.
    """

    row = dict(record)
    for name in ("time_utc", "event_anchor_time_utc", "observation_time_utc"):
        if row.get(name):
            row[name] = utc_text(row[name])
    edge_names = (
        ("south", "llcrnrlat"), ("west", "llcrnrlon"),
        ("north", "urcrnrlat"), ("east", "urcrnrlon"),
    )
    for box_name in ("bounding_box", "latlon_bbox"):
        box = record.get(box_name)
        edges = _edges(box)
        for (canonical, alias), edge in zip(edge_names, edges):
            if edge is None:
                continue
            existing_alias = optional_number(row.get(alias))
            if existing_alias is not None and existing_alias != edge:
                raise DataFrameSchemaError(
                    f"catalog bounding-box conflict: {box_name}.{canonical}={edge!r} "
                    f"but {alias}={existing_alias!r}"
                )
            row.setdefault(alias, edge)
    if not row.get("observation_status"):
        row["observation_status"] = str(
            "cached" if record.get("observation_cache_available") else
            "local" if record.get("local_hdf5_available", record.get("local_data_available", False)) else
            "remote" if record.get("remote_hdf5_available") else "unavailable"
        )
    return row


def event_value(raw: dict[str, Any]) -> Event:
    """Convert raw event metadata to the flat public ``Event`` record.

    Parameters
    ----------
    raw : dict
        Backend event payload.
    Returns
    -------
    Event
        Flat event metadata.
    """

    record = dict(raw["record"])
    offsets = np.asarray(record.get("minute_offsets") or [], dtype=np.int32)
    offsets.setflags(write=False)
    south, west, north, east = _edges(record.get("bounding_box"))
    return Event(
        subject_key=str(record.get("subject_key") or "").lower(),
        id=str(record.get("id") or "").upper(),
        event_anchor_time_utc=utc_text(record.get("time_utc")),
        event_type=str(record.get("event_type") or "Unlabeled"),
        frame_times_utc=tuple(utc_text(item) for item in record.get("frame_times_utc", [])),
        minute_offsets=offsets,
        south=south, west=west, north=north, east=east,
        spatial_metadata=dict(record.get("spatial_metadata") or {}),
    )


def observation_value(raw: dict[str, Any], *, channel: str = "vil") -> xr.DataArray:
    """Convert a SEVIR channel cube with one UTC coordinate convention.

    Parameters
    ----------
    raw : dict
        Backend observation payload.
    channel : str, default 'vil'
        Product channel; controls units and the default missing-value code.
    Returns
    -------
    xarray.DataArray
        Channel values with a datetime64 UTC index and relative-minute coordinate.
    """

    values = np.asarray(raw.get("values", raw.get("frames")), dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"{channel} did not return a time-row-column cube")
    missing = float(raw.get("missing_value", 255 if channel == "vil" else -9999))
    values = values.copy()
    values[values == missing] = np.nan
    minutes = np.asarray(raw.get("relative_minutes", raw.get("minute_offsets", [])), dtype=np.int32)
    times = tuple(utc_text(item) for item in raw.get("frame_times_utc", []))
    if len(minutes) != values.shape[0] or len(times) != values.shape[0]:
        raise ValueError(f"{channel} returned an inconsistent observation time axis")
    data = xr.DataArray(
        values,
        dims=("time", "row", "column"),
        coords={
            "time": ("time", pd.to_datetime(list(times), utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]")),
            "time_utc": ("time", np.asarray(times, dtype=str)),
            "minute_offset": ("time", minutes),
            "row": np.arange(values.shape[1], dtype=np.int32),
            "column": np.arange(values.shape[2], dtype=np.int32),
        },
        name=channel,
        attrs={
            "subject_key": str(raw.get("subject_key") or "").lower(),
            "event_id": str(raw.get("event_id") or "").upper(),
            "event_anchor_time_utc": utc_text(raw.get("event_anchor_time_utc")),
            "channel": channel,
            "units": {
                "vil": "encoded_vil", "vis": "scaled_reflectance",
                "ir069": "scaled_brightness_temperature", "ir107": "scaled_brightness_temperature",
                "lght": "flash_extent_density",
            }[channel],
            "missing_value": missing,
            "source": str(raw.get("data_source") or raw.get("source_hdf5") or "SEVIR"),
            "spatial_metadata": dict(raw.get("spatial_metadata") or {}),
        },
    )
    data.coords["time"].attrs["timezone"] = "UTC"
    if raw.get("time_selection") is not None:
        data.attrs["time_selection"] = dict(raw["time_selection"])
    if raw.get("requested_region") is not None:
        from .geospatial.projection import select_region_value
        data = select_region_value(data, raw["requested_region"])
    data.attrs["availability"] = {
        "scope": "returned_frames_and_selected_grid_cells",
        "valid_observation_cells": int(np.isfinite(data.values).sum()),
        "pixel_availability": "loaded",
        "absence_proven": False,
    }
    return data


def upload_match_value(upload_id: str, raw: dict[str, Any]) -> UploadMatch:
    """Convert upload matching output to a small immutable record.

    Parameters
    ----------
    upload_id : str
        Upload identity.
    raw : dict
        Backend matching payload.
    Returns
    -------
    UploadMatch
        Normalized match result.
    """

    match = raw.get("event_match") if isinstance(raw.get("event_match"), dict) else {}
    return UploadMatch(
        upload_id=upload_id,
        subject_key=str(match.get("subject_key") or "").lower() or None,
        event_id=str(match.get("event_id") or "").upper() or None,
        confidence=optional_number(match.get("confidence")),
        method=str(match.get("method") or match.get("status") or "unmatched"),
    )


def candidate_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a stable candidate table for geocoding results.

    Parameters
    ----------
    records : list of dict
        Candidate rows.
    Returns
    -------
    pandas.DataFrame
        Candidate table with fixed columns.
    """

    return normalize_dataframe(records, GEO_PLACE_CANDIDATE)
