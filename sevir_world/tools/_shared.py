from __future__ import annotations

"""Small, single-purpose adapters at the public capability boundary."""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..capabilities.models import Event, Location, PixelLocation, UploadMatch


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
    """Convert one raw catalog record to one stable DataFrame row.

    Parameters
    ----------
    record : dict
        Raw catalog record.
    Returns
    -------
    dict
        Canonical row fields.
    """

    south, west, north, east = _edges(record.get("bounding_box") or record.get("latlon_bbox"))
    missing = optional_number(record.get("pct_missing"))
    return {
        "subject_key": str(record.get("subject_key") or "").lower(),
        "event_id": str(record.get("id") or record.get("event_id") or "").upper(),
        "event_anchor_time_utc": utc_text(record.get("time_utc")),
        "event_type": str(record.get("event_type") or "Unlabeled"),
        "south": south,
        "west": west,
        "north": north,
        "east": east,
        "distance_km": optional_number(record.get("distance_km", record.get("distance_to_footprint_km"))),
        "data_min": optional_number(record.get("data_min")),
        "data_max": optional_number(record.get("data_max")),
        "missing_fraction": None if missing is None else missing / 100.0,
        "observation_status": str(record.get("observation_status") or (
            "cached" if record.get("observation_cache_available") else
            "local" if record.get("local_hdf5_available", record.get("local_data_available", False)) else
            "remote" if record.get("remote_hdf5_available") else "unavailable"
        )),
        "history_span_minutes": optional_number(record.get("history_span_minutes")),
        "history_expected_frames": optional_number(record.get("history_expected_frames")),
        "history_available_frames": optional_number(record.get("history_available_frames")),
        "history_is_continuous": record.get("history_is_continuous"),
    }


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
    )


def observation_value(raw: dict[str, Any]) -> xr.DataArray:
    """Convert a raw SEVIR cube to a standard VIL ``DataArray``.

    Parameters
    ----------
    raw : dict
        Backend observation payload.
    Returns
    -------
    xarray.DataArray
        VIL values with named dimensions and coordinates.
    """

    values = np.asarray(raw.get("values", raw.get("frames")), dtype=np.float32)
    missing = float(raw.get("missing_value", 255))
    values = values.copy()
    values[values == missing] = np.nan
    minutes = np.asarray(raw.get("relative_minutes") or [], dtype=np.int32)
    times = tuple(utc_text(item) for item in raw.get("frame_times_utc", []))
    data = xr.DataArray(
        values,
        dims=("time", "row", "column"),
        coords={
            "time_utc": ("time", np.asarray(times, dtype=str)),
            "minute_offset": ("time", minutes),
            "row": np.arange(values.shape[1], dtype=np.int32),
            "column": np.arange(values.shape[2], dtype=np.int32),
        },
        name="vil",
        attrs={
            "subject_key": str(raw.get("subject_key") or "").lower(),
            "event_id": str(raw.get("event_id") or "").upper(),
            "event_anchor_time_utc": utc_text(raw.get("event_anchor_time_utc")),
            "units": "encoded_vil",
            "missing_value": 255,
            "source": "SEVIR",
        },
    )
    return data


def pixel_value(value: dict[str, Any]) -> PixelLocation:
    """Convert one raw pixel result to a flat ``PixelLocation``.

    Parameters
    ----------
    value : dict
        Backend pixel payload.
    Returns
    -------
    PixelLocation
        Flat pixel metadata.
    """

    return PixelLocation(
        row=int(value["row"]), column=int(value["column"]),
        latitude=float(value["latitude"]), longitude=float(value["longitude"]),
        projected_x=optional_number(value.get("projected_x_m")),
        projected_y=optional_number(value.get("projected_y_m")),
        crs=str(value.get("crs") or ""),
    )


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

    return pd.DataFrame(records, columns=["name", "latitude", "longitude", "category", "place_type"])
