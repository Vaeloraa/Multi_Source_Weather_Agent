"""Read, search and cache source-native SEVIR catalog observations.

This is the single storage implementation shared by Agent-facing SEVIR tools,
CasCast inference, retrieval, uploads, and service readiness checks.
"""

from __future__ import annotations

from weather_agent.errors import error_record, is_transient_transport_error

import base64
import binascii
import csv
import hashlib
import json
import math
import re
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from ...config import ROOT
from .config import (
    CATALOG_CSV,
    LOCAL_SEVIR_DATA_ROOT,
    SEVIR_EVENT_CACHE,
    SEVIR_REMOTE_BASE_URL,
    SEVIR_REMOTE_BLOCK_SIZE,
    SEVIR_REMOTE_ENABLED,
)
from ...cache.registry import (
    CacheRegistry,
    registry_entry_key,
    versioned_artifact_identity,
)

import h5py
import numpy as np
try:
    import fsspec
except ImportError:  # Local-only deployments do not need the optional remote reader.
    fsspec = None  # type: ignore[assignment]
from pyproj import CRS, Transformer, Geod

from ..geospatial.projection import (
    bounding_boxes_intersect,
    geodesic_distance_to_bounds_km,
    region_bounds,
)
from .._shared import require_utc_text, utc_text


SEVIR_EVENT_CACHE_NAMESPACE = "sevir.observation.event.v1"
SEVIR_EVENT_CACHE_PIPELINE_VERSION = "sevir-event-array-v2"
SEVIR_EVENT_CACHE_LEASE_SECONDS = 300.0
SEVIR_EVENT_CACHE_WAIT_SECONDS = 30.0
SEVIR_REMOTE_READ_ATTEMPTS = 2


def _catalog_number(value: Any) -> float | None:
    """Normalize an optional catalog scalar as ``float``.

    Parameters
    ----------
    value : Any
        Catalog value that may be blank or missing.

    Returns
    -------
    float or None
        Numeric value when available; otherwise ``None``.
    """
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _catalog_integer(value: Any) -> int | None:
    """Normalize an optional catalog scalar as ``int``.

    Parameters
    ----------
    value : Any
        Catalog value that may be blank or missing.

    Returns
    -------
    int or None
        Integer value when available; otherwise ``None``.
    """
    number = _catalog_number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _catalog_subject_key(record: dict[str, Any]) -> str:
    """Return one stable opaque identity for an exact SEVIR catalog row.

    Parameters
    ----------
    record : dict
        Normalized catalog record containing event, file and row identity.

    Returns
    -------
    str
        Stable public key that does not expose backend path details.
    """

    material = "\x00".join((
        str(record.get("id") or "").strip().upper(),
        str(record.get("file_name") or "").replace("\\", "/").strip().lower(),
        str(record.get("file_index") if record.get("file_index") is not None else ""),
    ))
    return "sevir:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _minute_offsets(value: Any) -> tuple[int, ...]:
    """Parse catalog minute offsets into an immutable integer sequence.

    Parameters
    ----------
    value : Any
        Serialized or iterable minute-offset representation.

    Returns
    -------
    tuple of int
        Parsed frame offsets relative to the event time.
    """
    raw = str(value or "").strip()
    if not raw:
        return ()
    try:
        return tuple(int(item) for item in raw.split(":"))
    except ValueError as error:
        raise ValueError(f"Invalid SEVIR minute_offsets: {raw}") from error


def _frame_times(time_utc: str, offsets: tuple[int, ...]) -> list[str]:
    """Construct ISO-8601 frame times from an event time and minute offsets.

    Parameters
    ----------
    time_utc : str
        Event reference time in UTC.
    offsets : tuple of int
        Relative frame minutes.

    Returns
    -------
    list of str
        UTC timestamps in the same order as ``offsets``.
    """
    if not time_utc or not offsets:
        return []
    reference = _parse_query_time(time_utc)
    if reference is None:
        return []
    return [
        (reference + timedelta(minutes=offset)).isoformat().replace("+00:00", "Z")
        for offset in offsets
    ]


def _observation_time_query(
    observation_time_utc: str | None, time_match: str, tolerance_seconds: float | None,
) -> tuple[datetime | None, float]:
    """Validate an absolute target and an explicitly bounded nearest policy."""
    if time_match not in {"exact", "nearest"}:
        raise ValueError("time_match must be exact or nearest")
    if observation_time_utc is None:
        if time_match != "exact" or tolerance_seconds is not None:
            raise ValueError("time matching requires observation_time_utc")
        return None, 0.0
    target = _parse_query_time(require_utc_text(observation_time_utc, "observation_time_utc"))
    if time_match == "exact":
        if tolerance_seconds is not None:
            raise ValueError("tolerance_seconds is only valid with time_match='nearest'")
        return target, 0.0
    tolerance = float(tolerance_seconds) if tolerance_seconds is not None else math.nan
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("nearest requires finite positive tolerance_seconds")
    return target, tolerance


def _match_observation_frame(
    record: dict[str, Any], target: datetime, tolerance_seconds: float,
) -> dict[str, Any] | None:
    """Match actual frame offsets, choosing the earlier frame on ties."""
    anchor = _parse_query_time(record.get("time_utc"))
    offsets = record.get("minute_offsets")
    if anchor is None or not offsets:
        return None
    requested_offset = (target - anchor).total_seconds()
    index = min(range(len(offsets)), key=lambda i: (
        abs(offsets[i] * 60 - requested_offset), offsets[i], i,
    ))
    delta = offsets[index] * 60 - requested_offset
    actual = utc_text((anchor + timedelta(minutes=offsets[index])).isoformat())
    accepted = abs(delta) <= tolerance_seconds
    return {
        "requested_time_utc": utc_text(target.isoformat()),
        "observation_time_utc": actual if accepted else None,
        "time_match_kind": ("exact" if delta == 0 else "nearest") if accepted else "unavailable",
        "time_delta_seconds": delta if accepted else None,
        "tolerance_seconds": tolerance_seconds,
        "frame_index": index if accepted else None,
        "nearest_available_time_utc": actual,
        "nearest_delta_seconds": delta,
        "available_start_time_utc": utc_text((anchor + timedelta(minutes=min(offsets))).isoformat()),
        "available_end_time_utc": utc_text((anchor + timedelta(minutes=max(offsets))).isoformat()),
        "available_frame_count": len(offsets),
    }


def _path_label(path: Path, base: Path = ROOT) -> str:
    """Return a portable display label for a path when possible.

    Parameters
    ----------
    path : Path
        Path to label.
    base : Path, default ROOT
        Preferred root used to make the label relative.

    Returns
    -------
    str
        Relative path below ``base`` or the original path string.
    """
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


@lru_cache(maxsize=1)
def _catalog_all_records() -> tuple[dict[str, Any], ...]:
    """Load and cache normalized records from the SEVIR catalog for all image types.

    Returns
    -------
    tuple of dict
        Immutable container of normalized catalog records for all channels.
    """
    if not CATALOG_CSV.exists():
        raise RuntimeError("SEVIR CATALOG.csv was not found.")
    records: list[dict[str, Any]] = []
    with CATALOG_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            img_type = str(row.get("img_type", "")).lower()
            if img_type not in {"vil", "vis", "ir069", "ir107", "lght"}:
                continue
            time_utc = str(row.get("time_utc", ""))
            records.append(
                {
                    "id": str(row.get("id", "")),
                    "file_name": str(row.get("file_name", "")),
                    "file_index": int(row["file_index"]) if row.get("file_index", "").isdigit() else None,
                    "img_type": img_type,
                    "time_utc": time_utc,
                    "minute_offsets": _minute_offsets(row.get("minute_offsets")),
                    # Preserve the source catalog names as well as the normalized
                    # NOAA aliases used by older tools. This keeps schema discovery
                    # complete without breaking existing callers.
                    "episode_id": _catalog_integer(row.get("episode_id")),
                    "event_id": _catalog_integer(row.get("event_id")),
                    "year": int(time_utc[:4]) if len(time_utc) >= 4 and time_utc[:4].isdigit() else None,
                    "month": int(time_utc[5:7]) if len(time_utc) >= 7 and time_utc[5:7].isdigit() else None,
                    "noaa_episode_id": _catalog_integer(row.get("episode_id")),
                    "noaa_event_id": _catalog_integer(row.get("event_id")),
                    "event_type": str(row.get("event_type") or "Unlabeled"),
                    "llcrnrlat": _catalog_number(row.get("llcrnrlat")),
                    "llcrnrlon": _catalog_number(row.get("llcrnrlon")),
                    "urcrnrlat": _catalog_number(row.get("urcrnrlat")),
                    "urcrnrlon": _catalog_number(row.get("urcrnrlon")),
                    "proj": str(row.get("proj") or "").strip(),
                    "size_x": _catalog_integer(row.get("size_x")),
                    "size_y": _catalog_integer(row.get("size_y")),
                    "height_m": _catalog_number(row.get("height_m")),
                    "width_m": _catalog_number(row.get("width_m")),
                    "data_min": _catalog_number(row.get("data_min")),
                    "data_max": _catalog_number(row.get("data_max")),
                    "pct_missing": _catalog_number(row.get("pct_missing")),
                }
            )
    if not records:
        raise RuntimeError("No records were found in the SEVIR catalog.")
    return tuple(records)


@lru_cache(maxsize=1)
def _catalog_vil_records() -> tuple[dict[str, Any], ...]:
    """Load and cache normalized VIL records from the SEVIR catalog.

    Returns
    -------
    tuple of dict
        Immutable container of normalized VIL catalog records only.
    """
    return tuple(r for r in _catalog_all_records() if r["img_type"] == "vil")


def _catalog_public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Select stable public fields from an internal catalog record.

    Parameters
    ----------
    record : dict
        Normalized internal catalog record.

    Returns
    -------
    dict
        JSON-friendly event metadata safe for capability results.
    """
    public = {
        key: record.get(key)
        for key in (
            "id", "time_utc", "event_type", "file_name", "file_index",
            "data_min", "data_max", "pct_missing",
        )
    }
    public["subject_key"] = _catalog_subject_key(record)
    public["local_hdf5_available"] = (LOCAL_SEVIR_DATA_ROOT / record["file_name"]).exists()
    public["remote_hdf5_available"] = bool(SEVIR_REMOTE_ENABLED)
    # Use the same cache identity as the observation loader. File presence is
    # only a candidate for a cache hit; loading still validates its contents.
    public["observation_cache_available"] = _event_cache_path(record).is_file()
    if None not in (
        record.get("llcrnrlat"), record.get("llcrnrlon"),
        record.get("urcrnrlat"), record.get("urcrnrlon"),
    ):
        public["bounding_box"] = {
            "south": float(record["llcrnrlat"]),
            "west": float(record["llcrnrlon"]),
            "north": float(record["urcrnrlat"]),
            "east": float(record["urcrnrlon"]),
        }
    return public


@lru_cache(maxsize=1)
def _catalog_vil_index() -> dict[str, tuple[dict[str, Any], ...]]:
    """Build and cache an event-ID index over normalized VIL records.

    Returns
    -------
    dict
        Uppercase event IDs mapped to one or more catalog records.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in _catalog_vil_records():
        key = record["id"].lower()
        grouped.setdefault(key, []).append(record)
    return {key: tuple(records) for key, records in grouped.items()}


@lru_cache(maxsize=1)
def _catalog_subject_index() -> dict[str, dict[str, Any]]:
    """Index exact catalog rows by their opaque public subject key.

    Returns
    -------
    dict
        Lowercase subject keys mapped to normalized catalog records.
    """

    return {
        _catalog_subject_key(record).lower(): record
        for record in _catalog_vil_records()
    }


def _catalog_event_by_type(
    event_id: str,
    img_type: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Select one catalog record for an event, image type, and optional file identity.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    img_type : str
        Image type: 'vil', 'vis', 'ir069', 'ir107', or 'lght'.
    file_name : str, optional
        Exact source file name used to disambiguate duplicate IDs.
    file_index : int, optional
        Dataset index used to disambiguate duplicate IDs.

    Returns
    -------
    dict
        Matching normalized catalog record.

    Raises
    ------
    KeyError
        If no matching record exists.
    """
    target = str(event_id or "").strip().lower()
    img_type = str(img_type or "").strip().lower()
    if not target:
        raise ValueError("event_id is required.")
    if img_type not in {"vil", "vis", "ir069", "ir107", "lght"}:
        raise ValueError(f"Unsupported img_type: {img_type}")

    if target.startswith("sevir:"):
        record = _catalog_subject_index().get(target)
        if record is None or (
            record.get("img_type") is not None and record.get("img_type") != img_type
        ):
            raise ValueError(f"SEVIR CATALOG does not contain {img_type} subject {event_id}.")
        return record

    # Use the typed index for VIL so duplicate event IDs are handled by the
    # same identity source used by public search and subject-key resolution.
    if img_type == "vil":
        candidates = list(_catalog_vil_index().get(target, ()))
    else:
        candidates = [
            r for r in _catalog_all_records()
            if r["id"].lower() == target and r["img_type"] == img_type
        ]

    if file_name:
        normalized_file = str(file_name).replace("\\", "/").lower()
        candidates = [
            record for record in candidates
            if record["file_name"].replace("\\", "/").lower() == normalized_file
        ]
    if file_index is not None:
        candidates = [record for record in candidates if record["file_index"] == int(file_index)]
    if not candidates:
        raise ValueError(f"SEVIR CATALOG does not contain {img_type} event {event_id}.")
    if len(candidates) > 1:
        choices = [
            {"file_name": item["file_name"], "file_index": item["file_index"]}
            for item in candidates
        ]
        raise ValueError(
            f"SEVIR {img_type} id {event_id} has {len(candidates)} CATALOG records; "
            f"provide file_name or file_index. Candidates: {choices}"
        )
    return candidates[0]


def _catalog_vil_event(
    event_id: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Select one catalog record for a VIL event and optional file identity.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    file_name : str, optional
        Exact source file name used to disambiguate duplicate IDs.
    file_index : int, optional
        Dataset index used to disambiguate duplicate IDs.

    Returns
    -------
    dict
        Matching normalized catalog record.

    Raises
    ------
    KeyError
        If no matching record exists.
    """
    return _catalog_event_by_type(event_id, "vil", file_name, file_index)


def catalog_subject_key(record: dict[str, Any]) -> str:
    """Return the stable public identity for an exact SEVIR catalog row."""
    return _catalog_subject_key(record)


def catalog_vil_event(
    event_id: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Return one unambiguous VIL catalog row for an ID or subject key."""
    return _catalog_vil_event(event_id, file_name, file_index)


def catalog_vil_records() -> tuple[dict[str, Any], ...]:
    """Return every normalized VIL row with exact identity and geography."""

    records = []
    for record in _catalog_vil_records():
        public = {**record}
        public.update({
            "subject_key": _catalog_subject_key(record),
            "event_id": str(record.get("id") or ""),
            "time": str(record.get("time_utc") or ""),
            "bounding_box": {
                "south": record.get("llcrnrlat"),
                "west": record.get("llcrnrlon"),
                "north": record.get("urcrnrlat"),
                "east": record.get("urcrnrlon"),
            },
        })
        public["geographic_extent"] = dict(public["bounding_box"])
        records.append(public)
    return tuple(records)


def portable_path_label(path: Path, base: Path = ROOT) -> str:
    """Return a portable path label without exposing repository internals."""
    return _path_label(path, base)


def resolve_sevir_subject_key(identifier: str) -> str:
    """Resolve a unique catalog subject key from a key or event identifier.

    Parameters
    ----------
    identifier : str
        Existing ``subject_key`` or a unique human-readable event ID.

    Returns
    -------
    str
        Stable opaque subject key for the exact catalog row.
    """

    return _catalog_subject_key(_catalog_vil_event(identifier))


def _full_catalog_record(record: dict[str, Any]) -> dict[str, Any]:
    """Enrich a normalized catalog record with timing and availability fields.

    Parameters
    ----------
    record : dict
        Normalized internal catalog record.

    Returns
    -------
    dict
        Complete metadata record used by event-loading functions.
    """
    offsets = record["minute_offsets"]
    return {
        **_catalog_public_record(record),
        "sevir_id": record["id"],
        "img_type": record["img_type"],
        "minute_offsets": list(offsets),
        "frame_times_utc": _frame_times(record["time_utc"], offsets),
        "noaa_episode_id": record["noaa_episode_id"],
        "noaa_event_id": record["noaa_event_id"],
        "bounding_box": {
            "south": record["llcrnrlat"],
            "west": record["llcrnrlon"],
            "north": record["urcrnrlat"],
            "east": record["urcrnrlon"],
        },
        "proj": record["proj"],
        "size_x": record["size_x"],
        "size_y": record["size_y"],
        "height_m": record["height_m"],
        "width_m": record["width_m"],
        "spatial_metadata": spatial_metadata_for_record(record),
    }


def spatial_metadata_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Describe a catalog raster using the shared geospatial metadata schema.

    This provider performs no public coordinate conversion. It only translates
    source-native catalog fields into the metadata consumed by Geospatial.
    """

    required = (
        "proj", "size_x", "size_y", "width_m", "height_m",
        "llcrnrlon", "llcrnrlat", "urcrnrlon", "urcrnrlat",
    )
    missing = [name for name in required if record.get(name) in (None, "")]
    if missing:
        raise ValueError(f"raster spatial metadata is incomplete: {', '.join(missing)}")
    crs = CRS.from_user_input(str(record["proj"]))
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xmin, _ = transformer.transform(float(record["llcrnrlon"]), float(record["llcrnrlat"]))
    _, ymax = transformer.transform(float(record["urcrnrlon"]), float(record["urcrnrlat"]))
    width = float(record["width_m"])
    height = float(record["height_m"])
    return {
        "grid_type": "projected_raster",
        "crs": crs.to_string(),
        "shape": [int(record["size_y"]), int(record["size_x"])],
        "extent": {
            "xmin": float(xmin), "ymin": float(ymax) - height,
            "xmax": float(xmin) + width, "ymax": float(ymax),
        },
        "bounds": {
            "south": float(record["llcrnrlat"]), "west": float(record["llcrnrlon"]),
            "north": float(record["urcrnrlat"]), "east": float(record["urcrnrlon"]),
        },
        "axis_order": ["row", "column"],
        "row_direction": "north_to_south",
        "pixel_anchor": "center",
    }


def get_sevir_event_metadata(
    event_id: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Return complete metadata for one indexed SEVIR event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    file_name : str, optional
        Optional source-file discriminator.
    file_index : int, optional
        Optional dataset-index discriminator.

    Returns
    -------
    dict
        Event identity, timing, raster geometry and data availability.

    Examples
    --------
    Read metadata for the event selected by the caller::

        metadata = get_sevir_event_metadata(event_id)
    """
    record = _catalog_vil_event(event_id, file_name=file_name, file_index=file_index)
    return {
        "status": "ok",
        "task": "sevir_event_metadata",
        "data_source": _path_label(CATALOG_CSV),
        "record": _full_catalog_record(record),
        "interpretation_limit": (
            "This is one exact VIL metadata record. Pixel values are only loaded when "
            "load_sevir_vil_event is called and the referenced HDF5 file is available locally."
        ),
    }


def _local_hdf5_path(record: dict[str, Any]) -> Path:
    """Resolve the configured local HDF5 path for a catalog record.

    Parameters
    ----------
    record : dict
        Catalog record containing the source filename.

    Returns
    -------
    Path
        Expected local path to the SEVIR data file.
    """
    root = LOCAL_SEVIR_DATA_ROOT.resolve()
    path = (root / record["file_name"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("CATALOG file_name resolves outside the SEVIR data root.") from error
    return path


def _event_cache_path(record: dict[str, Any]) -> Path:
    """Return a stable cache path without trusting an event ID as a filename.

    Parameters
    ----------
    record : dict
        Catalog record containing event and source-file identity.

    Returns
    -------
    Path
        Sanitized local cache path for the event array.
    """
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["id"]))
    img_type = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(record.get("img_type") or "vil").lower()
    )
    # Preserve the established VIL filename because observation workflows
    # and the existing 242-file observation cache discover it by event ID.
    # Other channels need an explicit discriminator: the real catalog contains
    # more than ten thousand cross-channel (event_id, file_index) collisions.
    if img_type == "vil":
        return SEVIR_EVENT_CACHE / f"{safe_id}_{int(record['file_index'])}.npz"
    return SEVIR_EVENT_CACHE / f"{safe_id}_{img_type}_{int(record['file_index'])}.npz"




def _remote_hdf5_url(record: dict[str, Any]) -> str:
    """Construct the public object-store URL for a catalog data file.

    Parameters
    ----------
    record : dict
        Catalog record containing the source filename.

    Returns
    -------
    str
        Remote HDF5 URL.
    """
    relative = str(record["file_name"]).replace("\\", "/").lstrip("/")
    return f"{SEVIR_REMOTE_BASE_URL}/{quote(relative, safe='/')}"


def _event_cache_registry() -> CacheRegistry:
    """Return the shared metadata registry used by SEVIR event arrays.

    Returns
    -------
    CacheRegistry
        Process-safe registry backed by the configured SQLite database.
    """

    return CacheRegistry()


def _event_source_fingerprint(
    record: dict[str, Any], *, local_path: Path | None
) -> dict[str, Any]:
    """Build a portable fingerprint for one exact catalog source.

    The fingerprint deliberately excludes resolved local paths.  A local HDF5
    source is invalidated by size/mtime, while a remote source is tied to the
    public object URL and the exact normalized catalog row.  SEVIR public
    objects do not currently expose an ETag, so the row digest is the strongest
    cheap remote fingerprint available without another network request.

    Parameters
    ----------
    record : dict
        Exact normalized catalog row.
    local_path : pathlib.Path or None
        Resolved local HDF5 path when the source is local.

    Returns
    -------
    dict
        Portable source facts used by the versioned artifact identity.
    """

    row = {
        "subject_key": _catalog_subject_key(record),
        "event_id": str(record.get("id") or ""),
        "img_type": str(record.get("img_type") or "vil").lower(),
        "file_name": str(record.get("file_name") or "").replace("\\", "/"),
        "file_index": record.get("file_index"),
        "minute_offsets": [int(value) for value in record.get("minute_offsets", ())],
    }
    row_digest = hashlib.sha256(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if local_path is not None and local_path.is_file():
        stat = local_path.stat()
        source = {
            "kind": "local_hdf5",
            "file_name": row["file_name"],
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    else:
        source = {
            "kind": "remote_hdf5",
            "object_url": _remote_hdf5_url(record),
        }
    return {
        "dataset": "SEVIR",
        "catalog_row_sha256": row_digest,
        "record": row,
        "source": source,
    }


def _event_cache_identity(
    record: dict[str, Any], *, local_path: Path | None
) -> dict[str, Any]:
    """Return the versioned identity for an extracted SEVIR event cube.

    Parameters
    ----------
    record : dict
        Exact normalized catalog row.
    local_path : pathlib.Path or None
        Resolved local HDF5 path when the source is local.

    Returns
    -------
    dict
        Canonical identity and its stable request key.
    """

    return versioned_artifact_identity(
        SEVIR_EVENT_CACHE_NAMESPACE,
        source=_event_source_fingerprint(record, local_path=local_path),
        parameters={
            "layout": ["time", "row", "column"],
            "frame_count": len(record.get("minute_offsets", ())),
            "native_encoded_units": True,
        },
        pipeline_version=SEVIR_EVENT_CACHE_PIPELINE_VERSION,
    )


def _event_cache_source_label(path: Path) -> str:
    """Return a cache provenance label without exposing a local directory.

    Parameters
    ----------
    path : pathlib.Path
        Internal artifact path.

    Returns
    -------
    str
        Portable cache label containing only the artifact filename.
    """

    return f"cache:sevir_event/{path.name}"


def _local_hdf5_source_label(record: dict[str, Any]) -> str:
    """Return a portable provenance label for a configured local HDF5 file.

    Parameters
    ----------
    record : dict
        Exact normalized catalog row.

    Returns
    -------
    str
        Dataset-relative source label.
    """

    relative = str(record.get("file_name") or "").replace("\\", "/").lstrip("/")
    return f"local_sevir_hdf5:{relative}"


def _cached_scalar(payload: Any, name: str, default: Any = None) -> Any:
    """Read one optional scalar from an ``np.load`` payload.

    Parameters
    ----------
    payload : Any
        Open NumPy archive.
    name : str
        Scalar field name.
    default : Any, optional
        Value returned when the scalar is absent or not zero-dimensional.

    Returns
    -------
    Any
        Decoded scalar or ``default``.
    """

    if name not in payload.files:
        return default
    value = np.asarray(payload[name])
    return value.item() if value.ndim == 0 else default


def _read_event_cache(
    path: Path,
    record: dict[str, Any],
    identity: dict[str, Any],
    *,
    source_locator: str,
) -> np.ndarray | None:
    """Validate and read one versioned event NPZ.

    Parameters
    ----------
    path : pathlib.Path
        Candidate NPZ artifact.
    record : dict
        Exact normalized catalog row expected in the artifact.
    identity : dict
        Current versioned cache identity.
    source_locator : str
        Portable relative source name or public remote URL.

    Returns
    -------
    numpy.ndarray or None
        Validated event cube, or ``None`` on mismatch.
    """

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cached_id = str(_cached_scalar(payload, "event_id", ""))
            cached_subject = str(_cached_scalar(payload, "subject_key", ""))
            cached_index = _cached_scalar(payload, "file_index")
            cached_type = str(_cached_scalar(payload, "img_type", "vil")).lower()
            cached_request_key = str(
                _cached_scalar(payload, "cache_request_key", "")
            )
            cached_locator = str(
                _cached_scalar(
                    payload,
                    "source_locator",
                    _cached_scalar(payload, "source_url", ""),
                )
            )
            expected_subject = _catalog_subject_key(record)
            expected_type = str(record.get("img_type") or "vil").lower()
            exact_row = (
                cached_id == str(record.get("id") or "")
                and cached_type == expected_type
                and (
                    cached_index == record.get("file_index")
                    or (
                        cached_index is None
                        and cached_subject
                        and cached_subject == expected_subject
                    )
                )
                and cached_subject == expected_subject
            )
            if not exact_row:
                return None
            if cached_request_key != identity["request_key"]:
                return None
            if source_locator and cached_locator != source_locator:
                return None
            if "frames" not in payload.files:
                return None
            frames = np.asarray(payload["frames"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    expected_frames = len(record.get("minute_offsets", ()))
    if frames.ndim != 3:
        return None
    if expected_frames and frames.shape[0] != expected_frames:
        if frames.shape[-1] != expected_frames:
            return None
        frames = np.moveaxis(frames, -1, 0)
    return np.ascontiguousarray(frames)


def _register_event_cache(
    registry: CacheRegistry,
    record: dict[str, Any],
    identity: dict[str, Any],
    path: Path,
    *,
    source_kind: str,
) -> dict[str, Any]:
    """Register one validated event artifact with path-free public metadata.

    Parameters
    ----------
    registry : CacheRegistry
        Shared cache metadata registry.
    record : dict
        Exact normalized catalog row.
    identity : dict
        Current versioned cache identity.
    path : pathlib.Path
        Validated filesystem artifact.
    source_kind : str
        ``local_hdf5`` or ``remote_hdf5``.

    Returns
    -------
    dict
        Persisted registry entry.
    """

    return registry.register(
        SEVIR_EVENT_CACHE_NAMESPACE,
        identity,
        path,
        metadata={
            "dataset": "SEVIR",
            "artifact_role": "encoded_event_cube",
            "artifact_name": path.name,
            "subject_key": _catalog_subject_key(record),
            "event_id": str(record.get("id") or ""),
            "img_type": str(record.get("img_type") or "vil").lower(),
            "source_kind": source_kind,
        },
    )


def _resolve_event_cache(
    registry: CacheRegistry,
    record: dict[str, Any],
    identity: dict[str, Any],
    path: Path,
    *,
    source_kind: str,
    source_locator: str,
) -> np.ndarray | None:
    """Resolve a registered artifact or recover one canonical versioned NPZ.

    Parameters
    ----------
    registry : CacheRegistry
        Shared cache metadata registry.
    record : dict
        Exact normalized catalog row.
    identity : dict
        Current versioned cache identity.
    path : pathlib.Path
        Canonical artifact path.
    source_kind : str
        ``local_hdf5`` or ``remote_hdf5``.
    source_locator : str
        Portable relative source name or public remote URL.

    Returns
    -------
    numpy.ndarray or None
        Validated event cube when a cache is reusable.
    """

    entry = registry.resolve(SEVIR_EVENT_CACHE_NAMESPACE, identity)
    if entry is not None:
        registered_path = Path(str(entry.get("artifact_uri") or ""))
        try:
            same_artifact = registered_path.resolve() == path.resolve()
        except OSError:
            same_artifact = False
        if same_artifact:
            loaded = _read_event_cache(
                registered_path, record, identity, source_locator=source_locator
            )
            if loaded is not None:
                return loaded

    loaded = _read_event_cache(path, record, identity, source_locator=source_locator)
    if loaded is None:
        return None
    _register_event_cache(
        registry,
        record,
        identity,
        path,
        source_kind=source_kind,
    )
    # Count a validated registry recovery as a cache hit in the same registry
    # used by every other scientific artifact.
    registry.resolve(SEVIR_EVENT_CACHE_NAMESPACE, identity)
    return loaded


def _write_event_cache(
    path: Path,
    record: dict[str, Any],
    identity: dict[str, Any],
    values: np.ndarray,
    *,
    source_locator: str,
) -> None:
    """Atomically publish one self-describing event NPZ.

    Parameters
    ----------
    path : pathlib.Path
        Canonical destination path.
    record : dict
        Exact normalized catalog row.
    identity : dict
        Current versioned cache identity.
    values : numpy.ndarray
        Three-dimensional encoded event cube.
    source_locator : str
        Portable relative source name or public remote URL.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp.npz")
    source_url = source_locator if source_locator.startswith(("http://", "https://")) else ""
    try:
        np.savez_compressed(
            temporary,
            cache_schema_version=np.asarray(SEVIR_EVENT_CACHE_PIPELINE_VERSION),
            cache_request_key=np.asarray(identity["request_key"]),
            event_id=np.asarray(record["id"]),
            img_type=np.asarray(str(record.get("img_type") or "vil").lower()),
            subject_key=np.asarray(_catalog_subject_key(record)),
            source_locator=np.asarray(source_locator),
            source_url=np.asarray(source_url),
            source_fingerprint=np.asarray(
                json.dumps(
                    identity.get("canonical", {}).get("source", {}),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            file_index=np.asarray(record["file_index"]),
            frames=np.ascontiguousarray(values),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_event_cache(
    registry: CacheRegistry,
    record: dict[str, Any],
    identity: dict[str, Any],
    path: Path,
    *,
    source_kind: str,
    source_locator: str,
) -> np.ndarray | None:
    """Wait briefly for the process holding the cross-process build lease.

    Parameters
    ----------
    registry : CacheRegistry
        Shared cache metadata registry.
    record : dict
        Exact normalized catalog row.
    identity : dict
        Current versioned cache identity.
    path : pathlib.Path
        Canonical artifact path.
    source_kind : str
        ``local_hdf5`` or ``remote_hdf5``.
    source_locator : str
        Portable relative source name or public remote URL.

    Returns
    -------
    numpy.ndarray or None
        Event cube published by the lease owner, if it becomes available.
    """

    deadline = time.monotonic() + SEVIR_EVENT_CACHE_WAIT_SECONDS
    while time.monotonic() < deadline:
        values = _resolve_event_cache(
            registry,
            record,
            identity,
            path,
            source_kind=source_kind,
            source_locator=source_locator,
        )
        if values is not None:
            return values
        time.sleep(0.05)
    return None


def _load_or_build_event_cache(
    record: dict[str, Any],
    *,
    local_path: Path | None,
    source_kind: str,
    source_locator: str,
    source_label: str,
    loader: Callable[[], np.ndarray],
) -> tuple[np.ndarray, str]:
    """Resolve or single-flight build one extracted event cube.

    Parameters
    ----------
    record : dict
        Exact normalized catalog row.
    local_path : pathlib.Path or None
        Local HDF5 source used for file-stat invalidation, when present.
    source_kind : str
        ``local_hdf5`` or ``remote_hdf5``.
    source_locator : str
        Portable relative source name or public remote URL.
    source_label : str
        Path-free provenance returned on a cold read.
    loader : callable
        Deferred source reader invoked only by the lease owner.

    Returns
    -------
    tuple of numpy.ndarray and str
        Loaded event cube and path-free provenance label.
    """

    cache_path = _event_cache_path(record)
    identity = _event_cache_identity(record, local_path=local_path)
    registry = _event_cache_registry()
    cached = _resolve_event_cache(
        registry,
        record,
        identity,
        cache_path,
        source_kind=source_kind,
        source_locator=source_locator,
    )
    if cached is not None:
        return cached, _event_cache_source_label(cache_path)

    lease_key = registry_entry_key(SEVIR_EVENT_CACHE_NAMESPACE, identity)
    owner = uuid.uuid4().hex
    acquired = registry.acquire_lease(
        lease_key,
        owner,
        lease_seconds=SEVIR_EVENT_CACHE_LEASE_SECONDS,
    )
    if not acquired:
        cached = _wait_for_event_cache(
            registry,
            record,
            identity,
            cache_path,
            source_kind=source_kind,
            source_locator=source_locator,
        )
        if cached is not None:
            return cached, _event_cache_source_label(cache_path)
        acquired = registry.acquire_lease(
            lease_key,
            owner,
            lease_seconds=SEVIR_EVENT_CACHE_LEASE_SECONDS,
        )
        if not acquired:
            raise RuntimeError(
                "SEVIR event cache is being prepared by another worker; retry this request."
            )

    try:
        cached = _resolve_event_cache(
            registry,
            record,
            identity,
            cache_path,
            source_kind=source_kind,
            source_locator=source_locator,
        )
        if cached is not None:
            return cached, _event_cache_source_label(cache_path)
        values = np.asarray(loader())
        if values.ndim != 3:
            raise ValueError(f"Expected a 3-D SEVIR event, got shape {values.shape}.")
        values = np.ascontiguousarray(values)
        _write_event_cache(
            cache_path,
            record,
            identity,
            values,
            source_locator=source_locator,
        )
        _register_event_cache(
            registry,
            record,
            identity,
            cache_path,
            source_kind=source_kind,
        )
        return values, source_label
    finally:
        registry.release_lease(lease_key, owner)


def _read_dataset_by_type(handle: Any, record: dict[str, Any]) -> np.ndarray:
    """Read one event cube from an open SEVIR HDF5 handle for any image type.

    Parameters
    ----------
    handle : Any
        Open ``h5py.File``-compatible handle.
    record : dict
        Catalog record identifying the dataset, image type, and file index.

    Returns
    -------
    np.ndarray
        Event frames in native encoded units.
    """
    img_type = record["img_type"]
    if img_type not in handle:
        raise ValueError(f"SEVIR HDF5 does not contain dataset '{img_type}'.")
    dataset = handle[img_type]
    file_index = record["file_index"]
    if file_index is None:
        raise ValueError(f"SEVIR event {record['id']} has no valid file_index.")
    if not 0 <= file_index < dataset.shape[0]:
        raise IndexError(
            f"file_index {file_index} is outside dataset '{img_type}' with {dataset.shape[0]} events."
        )
    sample = np.asarray(dataset[file_index])
    expected_frames = len(record["minute_offsets"])
    if sample.ndim != 3:
        raise ValueError(f"Expected a 3-D {img_type} event, got shape {sample.shape}.")
    time_axes = [axis for axis, size in enumerate(sample.shape) if size == expected_frames]
    if expected_frames and len(time_axes) == 1:
        sample = np.moveaxis(sample, time_axes[0], 0)
    elif sample.shape[-1] == 49:
        sample = np.moveaxis(sample, -1, 0)
    elif sample.shape[0] != 49:
        raise ValueError(f"Cannot identify the time axis in {img_type} shape {sample.shape}.")

    # Preserve original dtype for different channels
    if img_type == "vil":
        return np.ascontiguousarray(sample, dtype=np.uint8)
    elif img_type in ("ir069", "ir107"):
        return np.ascontiguousarray(sample, dtype=np.int16)
    elif img_type == "vis":
        return np.ascontiguousarray(sample, dtype=np.int16)
    elif img_type == "lght":
        return np.ascontiguousarray(sample, dtype=np.float32)
    else:
        return np.ascontiguousarray(sample)


def _read_vil_dataset(handle: Any, record: dict[str, Any]) -> np.ndarray:
    """Read one VIL event cube from an open SEVIR HDF5 handle.

    Parameters
    ----------
    handle : Any
        Open ``h5py.File``-compatible handle.
    record : dict
        Catalog record identifying the dataset and file index.

    Returns
    -------
    np.ndarray
        Event VIL frames in native encoded units.
    """
    return _read_dataset_by_type(handle, {**record, "img_type": "vil"})


def _load_remote_array_by_type(
    record: dict[str, Any], *, allow_remote: bool | None = None
) -> tuple[np.ndarray, str]:
    """Download or reuse a cached remote event array for any image type.

    Parameters
    ----------
    record : dict
        Catalog record identifying the remote event dataset and image type.

    Returns
    -------
    values : np.ndarray
        Loaded event cube.
    source : str
        Source or cache provenance label.
    """
    normalized = {**record, "img_type": str(record.get("img_type") or "vil").lower()}
    url = _remote_hdf5_url(normalized)

    def load_remote() -> np.ndarray:
        """Read the exact remote row only after cache and lease checks miss.

        Returns
        -------
        numpy.ndarray
            Extracted event cube in native encoded units.
        """

        remote_enabled = SEVIR_REMOTE_ENABLED if allow_remote is None else bool(allow_remote)
        if not remote_enabled:
            raise FileNotFoundError(
                f"The HDF5 file for {normalized['id']} ({normalized['img_type']}) "
                "is not local and remote SEVIR access is disabled."
            )
        if fsspec is None:
            raise RuntimeError(
                "Remote SEVIR access requires the optional 'fsspec' package; "
                "install fsspec or set SEVIR_REMOTE_ENABLED=0."
            )
        for attempt in range(1, SEVIR_REMOTE_READ_ATTEMPTS + 1):
            try:
                with fsspec.open(
                    url,
                    mode="rb",
                    block_size=SEVIR_REMOTE_BLOCK_SIZE,
                    cache_type="blockcache",
                ) as remote_file:
                    with h5py.File(remote_file, "r") as handle:
                        return _read_dataset_by_type(handle, normalized)
            except Exception as error:
                if not is_transient_transport_error(error) or attempt >= SEVIR_REMOTE_READ_ATTEMPTS:
                    raise
                time.sleep(0.2 * attempt)

    return _load_or_build_event_cache(
        normalized,
        local_path=None,
        source_kind="remote_hdf5",
        source_locator=url,
        source_label=f"remote:{url}",
        loader=load_remote,
    )


def _load_remote_vil_array(
    record: dict[str, Any], *, allow_remote: bool | None = None
) -> tuple[np.ndarray, str]:
    """Download or reuse a cached remote VIL event array.

    Parameters
    ----------
    record : dict
        Catalog record identifying the remote event dataset.

    Returns
    -------
    values : np.ndarray
        Loaded VIL event cube.
    source : str
        Source or cache provenance label.
    """
    values, source = _load_remote_array_by_type(
        {**record, "img_type": "vil"}, allow_remote=allow_remote
    )
    return np.asarray(values, dtype=np.uint8), source


def _load_vil_array(
    record: dict[str, Any], *, allow_remote: bool | None = None
) -> tuple[np.ndarray, str]:
    """Load a VIL event from local storage or the configured remote source.

    Parameters
    ----------
    record : dict
        Catalog record for the requested event.

    Returns
    -------
    values : np.ndarray
        Loaded VIL event cube.
    source : str
        Source provenance label.
    """
    return _load_array_by_type(
        {**record, "img_type": "vil"}, allow_remote=allow_remote
    )


def _load_array_by_type(
    record: dict[str, Any], *, allow_remote: bool | None = None
) -> tuple[np.ndarray, str]:
    """Load event array from local HDF5 or remote cache for any image type.

    Parameters
    ----------
    record : dict
        Catalog record with img_type specified.

    Returns
    -------
    values : np.ndarray
        Loaded event cube.
    source : str
        Source provenance label.
    """
    normalized = {**record, "img_type": str(record.get("img_type") or "vil").lower()}
    local_path = _local_hdf5_path(normalized)
    if local_path.is_file():
        def load_local() -> np.ndarray:
            """Read one local source row only after cache and lease checks miss.

            Returns
            -------
            numpy.ndarray
                Extracted event cube in native encoded units.
            """

            with h5py.File(local_path, "r") as handle:
                return _read_dataset_by_type(handle, normalized)

        return _load_or_build_event_cache(
            normalized,
            local_path=local_path,
            source_kind="local_hdf5",
            source_locator=str(normalized.get("file_name") or "").replace("\\", "/"),
            source_label=_local_hdf5_source_label(normalized),
            loader=load_local,
        )
    return _load_remote_array_by_type(normalized, allow_remote=allow_remote)


def load_sevir_vil_event(
    event_id: str,
    start_frame: int = 0,
    end_frame: int = 49,
    file_name: str = "",
    file_index: int | None = None,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
    allow_remote: bool | None = None,
    *,
    observation_time_utc: str | None = None,
    time_match: str = "exact",
    tolerance_seconds: float | None = None,
    region: Any | None = None,
) -> dict[str, Any]:
    """Load a bounded frame interval for one SEVIR VIL event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    start_frame : int, default 0
        Inclusive zero-based first frame.
    end_frame : int, default 49
        Exclusive zero-based last frame.
    file_name : str, optional
        Optional source-file discriminator.
    file_index : int, optional
        Optional dataset-index discriminator.

    Returns
    -------
    dict
        Frame array, relative minutes, dimensions and event provenance.

    Examples
    --------
    Load the first 13 frames for the event selected by the caller::

        cube = load_sevir_vil_event(event_id, 0, 13)
    """
    if region is not None:
        from ..geospatial.contracts import region_from_value
        region = region_from_value(region)
    record = _catalog_vil_event(event_id, file_name=file_name, file_index=file_index)
    target, tolerance = _observation_time_query(observation_time_utc, time_match, tolerance_seconds)
    if any(value is not None and int(value) < 0 for value in (minutes_before, minutes_after)):
        raise ValueError("minutes_before and minutes_after must be non-negative")
    selection = None
    if target is not None:
        if minutes_before is not None or minutes_after is not None or start_frame != 0 or end_frame != 49:
            raise ValueError("absolute observation time cannot be combined with relative/index windows")
        selection = _match_observation_frame(record, target, tolerance)
        if selection is None or selection["frame_index"] is None:
            return {
                "status": "data_unavailable", "code": "OBSERVATION_TIME_UNAVAILABLE",
                "reason": "No catalog frame satisfies the requested time policy for this subject.",
                "subject_key": _catalog_subject_key(record), "pixel_data_loaded": False,
                "time_selection": selection,
                "availability": {
                    "scope": "selected_subject_catalog_frames",
                    "requested_time_utc": require_utc_text(observation_time_utc, "observation_time_utc"),
                    "absence_proven": selection is not None,
                    "pixel_availability": "not_checked",
                },
                "retryable": False,
            }
    try:
        frames, source = _load_vil_array(record, allow_remote=allow_remote)
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "status": "data_unavailable",
            "task": "load_sevir_vil_event",
            "event_id": record["id"],
            "subject_key": _catalog_subject_key(record),
            "pixel_data_loaded": False,
            "required_hdf5": record["file_name"],
            "file_index": record["file_index"],
            "metadata_available": True,
            "spatial_metadata": spatial_metadata_for_record(record),
            "frame_times_utc": _frame_times(record["time_utc"], record["minute_offsets"]),
            "reason": str(error),
            "retryable": is_transient_transport_error(error),
            "error": error_record(error),
            "time_selection": selection,
            "availability": {
                "scope": "selected_subject_pixel_read", "absence_proven": False,
                "pixel_availability": "read_failed",
            },
        }
    if frames.ndim != 3 or frames.shape[0] != len(record["minute_offsets"]):
        raise ValueError("loaded pixels disagree with the catalog time axis; no substitute frame selected")
    if selection is not None:
        start = selection["frame_index"]
        end = start + 1
        if end > frames.shape[0]:
            raise ValueError("catalog time axis exceeds the loaded pixel array; no substitute frame selected")
    elif minutes_before is not None or minutes_after is not None:
        before = int(minutes_before or 0)
        after = int(minutes_after or 0)
        positions = [
            index for index, offset in enumerate(record["minute_offsets"])
            if -before <= int(offset) <= after
        ]
        if not positions:
            raise ValueError(
                f"No observations fall in the requested {-before}..{after} minute window."
            )
        start, end = positions[0], positions[-1] + 1
    else:
        start = max(0, int(start_frame))
        end = min(int(end_frame), int(frames.shape[0]))
    if start >= end:
        raise ValueError(f"Invalid frame interval [{start_frame}, {end_frame}).")
    selected = frames[start:end]
    offsets = record["minute_offsets"][start:end]
    times = _frame_times(record["time_utc"], offsets)
    return {
        "status": "ok",
        "task": "load_sevir_vil_event",
        "event_id": record["id"],
        "subject_key": _catalog_subject_key(record),
        "event_anchor_time_utc": record["time_utc"],
        "time_selection": selection,
        "requested_region": region,
        "frame_times_utc": times,
        "source_hdf5": source,
        "file_index": record["file_index"],
        "full_event_shape": [int(value) for value in frames.shape],
        "loaded_shape": [int(value) for value in selected.shape],
        "frame_interval": {"start_inclusive": start, "end_exclusive": end},
        # Full frames stay inside the isolated generated-code worker as a
        # read-only ndarray.  The worker/trace serializers replace large
        # arrays with shape metadata, so pixels are available for Agent-written
        # calculations without being copied into the LLM observation.
        "frames": np.ascontiguousarray(selected, dtype=np.float32),
        "values": np.ascontiguousarray(selected, dtype=np.float32),
        "relative_minutes": list(offsets),
        "spatial_metadata": spatial_metadata_for_record(record),
        "dimensions": ["time", "row", "column"],
        "missing_value": 255,
        "pixel_data_loaded": True,
        "pixel_data_returned_to_llm": False,
        "pixel_data_available_to_generated_code": True,
        "interpretation_limit": (
            "The HDF5 pixels were read for computation, but the full array is not embedded in the "
            "tool response. Local HDF5, remote HDF5, or an event NPZ cache may be used. "
            "VIL value 255 is treated as missing."
        ),
    }


def load_sevir_vil_events(
    event_ids: list[str] | tuple[str, ...],
    start_frame: int = 0,
    end_frame: int = 49,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
) -> dict[str, Any]:
    """Load several real SEVIR observation cubes through one public operation.

    Parameters
    ----------
    event_ids : list or tuple of str
        Unique SEVIR event identifiers, limited to 20.
    start_frame, end_frame : int
        Inclusive start and exclusive end frame indices.

    Returns
    -------
    dict
        Loaded records plus explicit unavailable-event records. Status is
        ``partial`` when at least one requested event cannot be loaded.

    This backend deliberately owns iteration and partial availability.  Agent
    code receives one batch value and therefore does not create an N+1 series
    of public tool calls after a catalog search.
    """

    requested = tuple(dict.fromkeys(
        str(value).strip().lower() for value in event_ids if str(value).strip()
    ))
    if not requested:
        raise ValueError("event_ids must contain at least one SEVIR event ID")
    if len(requested) > 20:
        raise ValueError("event_ids is limited to 20 events per batch")
    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for event_id in requested:
        try:
            loaded = load_sevir_vil_event(
                event_id, start_frame, end_frame,
                minutes_before=minutes_before,
                minutes_after=minutes_after,
            )
        except Exception as error:
            unavailable.append({"subject_key": event_id, "reason": str(error)})
            continue
        if str(loaded.get("status", "")).lower() == "ok":
            records.append(loaded)
        else:
            unavailable.append({
                "subject_key": event_id,
                "reason": str(loaded.get("reason") or loaded.get("status") or "unavailable"),
            })
    return {
        "status": "partial" if unavailable else "ok",
        "requested_event_ids": list(requested),
        "records": records,
        "unavailable": unavailable,
    }


def _parse_query_time(value: Any) -> datetime | None:
    """Parse an optional public query timestamp as aware UTC.

    Parameters
    ----------
    value : Any
        ISO-8601 or catalog-style timestamp.

    Returns
    -------
    datetime or None
        UTC-aware timestamp, or ``None`` when the value is blank/invalid.
    """
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc))


def discover_sevir_catalog_candidates(
    *,
    region: Any | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    max_distance_km: float | None = None,
    before_time_utc: str | None = None,
    exclude_subject_key: str = "",
    event_type: str = "",
    max_month_distance: int | None = None,
    require_prior_minutes: int = 60,
) -> dict[str, Any]:
    """Discover the complete spatially eligible VIL candidate set in Catalog.

    This operation is metadata-only.  It deliberately does not inspect the
    event cache or HDF5 availability, so storage state cannot define the
    historical corpus.
    """

    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be supplied together")
    point = None if latitude is None else (float(latitude), float(longitude))
    if point is not None and (
        not -90 <= point[0] <= 90 or not -180 <= point[1] <= 180
    ):
        raise ValueError("latitude/longitude is outside the WGS84 range")
    distance_limit = (
        None if point is None and max_distance_km is None
        else float(100.0 if max_distance_km is None else max_distance_km)
    )
    if distance_limit is not None and distance_limit <= 0:
        raise ValueError("max_distance_km must be positive")
    normalized_region = None if region is None else region_bounds(region).as_dict()
    cutoff = _parse_query_time(before_time_utc)
    reference_month = cutoff.month if cutoff is not None else None
    event_type_normalized = str(event_type or "").strip().lower()
    excluded_subject = str(exclude_subject_key or "").strip().lower()
    exclusions: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for record in catalog_vil_records():
        subject_key = str(record["subject_key"]).lower()
        if excluded_subject and subject_key == excluded_subject:
            exclude("reference_event")
            continue
        record_time = _parse_query_time(record.get("time_utc"))
        if cutoff is not None and (record_time is None or record_time >= cutoff):
            exclude("not_strictly_historical")
            continue
        if event_type_normalized and str(record.get("event_type") or "").lower() != event_type_normalized:
            exclude("different_event_type")
            continue
        if max_month_distance is not None:
            candidate_month = record.get("month")
            if reference_month is None or candidate_month is None:
                exclude("month_unavailable")
                continue
            difference = abs(int(candidate_month) - int(reference_month))
            if min(difference, 12 - difference) > int(max_month_distance):
                exclude("outside_month_window")
                continue
        offsets = {int(value) for value in record.get("minute_offsets", ())}
        required = set(range(-abs(int(require_prior_minutes)), 1, 5))
        if not required.issubset(offsets):
            exclude("insufficient_prior_history")
            continue
        footprint = record.get("bounding_box")
        if not isinstance(footprint, dict) or any(footprint.get(key) is None for key in ("south", "west", "north", "east")):
            exclude("geography_unavailable")
            continue
        if normalized_region is not None and not bounding_boxes_intersect(normalized_region, footprint):
            exclude("outside_region")
            continue
        distance_km = None
        if point is not None:
            distance_km = geodesic_distance_to_bounds_km(point[0], point[1], footprint)
            if distance_limit is not None and distance_km > distance_limit:
                exclude("outside_geographic_radius")
                continue
        item = dict(record)
        if distance_km is not None:
            item["distance_to_footprint_km"] = distance_km
        candidates.append(item)

    candidates.sort(key=lambda item: (
        float(item.get("distance_to_footprint_km", 0.0)),
        str(item.get("time_utc") or ""),
        str(item.get("subject_key") or ""),
    ))
    return {
        "status": "ok",
        "task": "sevir_catalog_candidate_discovery",
        "data_source": _path_label(CATALOG_CSV),
        "catalog_record_count": len(catalog_vil_records()),
        "catalog_candidate_count": len(candidates),
        "excluded_count": sum(exclusions.values()),
        "exclusions": exclusions,
        "filters": {
            "region": normalized_region,
            "latitude": None if point is None else point[0],
            "longitude": None if point is None else point[1],
            "max_distance_km": distance_limit,
            "before_time_utc_exclusive": before_time_utc,
            "event_type": event_type,
            "max_month_distance": max_month_distance,
            "require_prior_minutes": int(require_prior_minutes),
        },
        "records": candidates,
        "corpus_source": "SEVIR Catalog VIL rows; cache and HDF5 availability were not discovery filters",
    }


def _boxes_intersect(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two WGS84 bounding boxes overlap.

    Parameters
    ----------
    left, right : dict
        Boxes containing ``south``, ``west``, ``north`` and ``east``.

    Returns
    -------
    bool
        ``True`` when the closed rectangles intersect.
    """
    try:
        return bounding_boxes_intersect(left, right)
    except (KeyError, TypeError, ValueError):
        return False


def search_sevir_catalog(
    query: str = "",
    year: int | None = None,
    month: int | None = None,
    event_type: str = "",
    region: Any | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
    start_time_utc: str | None = None,
    end_time_utc: str | None = None,
    require_prior_minutes: int | None = None,
    limit: int = 10,
    *,
    observation_time_utc: str | None = None,
    time_match: str = "exact",
    tolerance_seconds: float | None = None,
) -> dict[str, Any]:
    """Search normalized SEVIR catalog metadata with optional filters.

    Parameters
    ----------
    query : str, optional
        Case-insensitive text matched against event identity and metadata.
    year, month : int, optional
        UTC temporal filters.
    event_type : str, optional
        Event-label filter such as ``"Hail"``.
    region : dict, optional
        WGS84 bounding box; catalog footprints must intersect it.
    latitude, longitude : float, optional
        WGS84 centre for a geodesic distance-to-footprint filter.
    radius_km : float, optional
        Maximum point-to-footprint distance. Defaults to 100 km when a point
        is supplied.
    start_time_utc, end_time_utc : str, optional
        Inclusive UTC event-anchor time window.
    require_prior_minutes : int, optional
        Require the sequence to contain an observation at or before this
        negative offset (for example ``60`` requires at least one hour prior).
    limit : int, default 10
        Maximum number of returned records.

    Returns
    -------
    dict
        Matching public records and query provenance.

    Examples
    --------
    Find up to five hail records from 2019::

        rows = search_sevir_catalog(year=2019, event_type="Hail", limit=5)
    """
    query_normalized = str(query or "").strip().lower()
    event_type_normalized = str(event_type or "").strip().lower()
    region = region_bounds(region).as_dict() if region is not None else None
    target, tolerance = _observation_time_query(observation_time_utc, time_match, tolerance_seconds)
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be supplied together")
    point: tuple[float, float] | None = None
    geod: Geod | None = None
    distance_limit_km: float | None = None
    if latitude is not None and longitude is not None:
        point = (float(latitude), float(longitude))
        if not -90 <= point[0] <= 90 or not -180 <= point[1] <= 180:
            raise ValueError("latitude/longitude is outside the WGS84 range")
        distance_limit_km = max(0.0, float(radius_km if radius_km is not None else 100.0))
        geod = Geod(ellps="WGS84")
    start_dt = _parse_query_time(require_utc_text(start_time_utc, "start_time_utc")) if start_time_utc is not None else None
    end_dt = _parse_query_time(require_utc_text(end_time_utc, "end_time_utc")) if end_time_utc is not None else None
    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError("start_time_utc must not be after end_time_utc")
    prior = int(require_prior_minutes) if require_prior_minutes is not None else None
    limit = max(1, min(int(limit), 50))
    candidates: list[tuple[dict[str, Any], float | None, dict[str, Any] | None]] = []
    indeterminate = 0
    records = _catalog_vil_records()
    for record in records:
        if query_normalized:
            searchable = " ".join(
                str(record.get(key, "")) for key in ("id", "file_name", "time_utc", "event_type")
            ).lower()
            if query_normalized not in searchable:
                continue
        if year is not None and record["year"] != int(year):
            continue
        if month is not None and record["month"] != int(month):
            continue
        if event_type_normalized and event_type_normalized not in record["event_type"].lower():
            continue
        record_dt = _parse_query_time(record["time_utc"])
        if (start_dt or end_dt) and record_dt is None:
            indeterminate += 1
            continue
        if start_dt and (record_dt is None or record_dt < start_dt):
            continue
        if end_dt and (record_dt is None or record_dt > end_dt):
            continue
        if prior is not None:
            if not record["minute_offsets"]:
                indeterminate += 1
                continue
            available_offsets = {int(offset) for offset in record["minute_offsets"]}
            required_offsets = set(range(-abs(prior), 1, 5))
            if not required_offsets.issubset(available_offsets):
                continue
        footprint = {
            "south": record["llcrnrlat"], "west": record["llcrnrlon"],
            "north": record["urcrnrlat"], "east": record["urcrnrlon"],
        }
        if (region or point is not None) and any(footprint[key] is None for key in footprint):
            indeterminate += 1
            continue
        if region or point is not None:
            try:
                region_bounds(footprint)
            except (KeyError, TypeError, ValueError):
                indeterminate += 1
                continue
        if region and not _boxes_intersect(region, footprint):
            continue
        distance_km: float | None = None
        if point is not None and geod is not None and distance_limit_km is not None:
            if any(footprint[key] is None for key in footprint):
                continue
            distance_km = geodesic_distance_to_bounds_km(
                point[0], point[1], footprint
            )
            if distance_km > distance_limit_km:
                continue
        selection = None
        if target is not None:
            selection = _match_observation_frame(record, target, tolerance)
            if selection is None:
                indeterminate += 1
                continue
            if selection["frame_index"] is None:
                continue
        candidates.append((record, distance_km, selection))
    # Rank the complete candidate set before truncating. Catalog order is
    # never evidence that a frame is the nearest one to the requested time.
    if target is not None:
        candidates.sort(key=lambda value: (
            abs(value[2]["time_delta_seconds"]), value[2]["observation_time_utc"],
            _catalog_subject_key(value[0]),
        ))
    total_matches = len(candidates)
    matches: list[dict[str, Any]] = []
    for record, distance_km, selection in candidates[:limit]:
        item = _catalog_public_record(record)
        offsets = sorted({int(value) for value in record["minute_offsets"]})
        required_offsets = list(range(-abs(prior), 1, 5)) if prior is not None else []
        item.update({
            "history_span_minutes": max(0, -min(offsets)) if offsets else 0,
            "history_expected_frames": len(required_offsets),
            "history_available_frames": sum(value in offsets for value in required_offsets),
            "history_is_continuous": (
                all(value in offsets for value in required_offsets) if required_offsets else None
            ),
        })
        if selection is not None:
            item.update({key: selection[key] for key in (
                "observation_time_utc", "time_delta_seconds", "time_match_kind",
            )})
        if distance_km is not None:
            item["distance_to_footprint_km"] = round(distance_km, 3)
        matches.append(item)
    catalog_stat = CATALOG_CSV.stat()
    return {
        "status": "ok",
        "task": "sevir_catalog_search",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "filters": {"query": query, "year": year, "month": month, "event_type": event_type,
                    "region": region, "latitude": latitude, "longitude": longitude,
                    "radius_km": distance_limit_km,
                    "start_time_utc": start_dt.isoformat().replace("+00:00", "Z") if start_dt else None,
                    "end_time_utc": end_dt.isoformat().replace("+00:00", "Z") if end_dt else None,
                    "require_prior_minutes": prior, "limit": limit,
                    "observation_time_utc": target.isoformat().replace("+00:00", "Z") if target else None,
                    "time_match": time_match, "tolerance_seconds": tolerance},
        "total_matches": total_matches,
        "returned_count": len(matches),
        "truncated": total_matches > len(matches),
        "units": "encoded_vil",
        "availability": {
            "scope": "filtered_catalog_frame_coverage" if target else "filtered_catalog_event_anchors",
            "catalog": str(CATALOG_CSV.relative_to(ROOT)),
            "catalog_size_bytes": catalog_stat.st_size,
            "catalog_mtime_ns": catalog_stat.st_mtime_ns,
            "catalog_vil_records": len(records),
            "scan_complete": True,
            "indeterminate_records": indeterminate,
            "absence_proven": target is not None and total_matches == 0 and indeterminate == 0,
            "pixel_availability": "not_checked",
            "spatial_match": "bounding_box_intersection" if region else "point_radius" if point else "unrestricted",
            "full_region_coverage_proven": False,
        },
        "records": matches,
        "interpretation_limit": "Catalog metadata returned. Retrieve HDF5 observations separately for pixel-level analysis.",
    }


def search_sevir_catalog_by_region(
    latitude: float,
    longitude: float,
    radius_km: float = 100.0,
    month: int | None = None,
    year: int | None = None,
    event_type: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Find catalog events whose footprint intersects a point window.

    Parameters
    ----------
    latitude, longitude : float
        Centre of the search window in decimal degrees.
    radius_km : float
        Search radius in kilometres.
    year, month : int, optional
        UTC temporal filters.
    event_type : str, optional
        Catalog event-label filter.
    limit : int
        Maximum records returned.

    Returns
    -------
    dict
        Metadata-only candidate records and search provenance.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside the WGS84 range")
    radius_km = max(0.0, float(radius_km))
    geod = Geod(ellps="WGS84")
    _, _, meters_per_lat_degree = geod.inv(longitude, latitude, longitude, min(89.999999, latitude + 1.0))
    _, _, meters_per_lon_degree = geod.inv(longitude, latitude, longitude + 1.0, latitude)
    lat_delta = radius_km * 1000.0 / max(1.0, meters_per_lat_degree)
    lon_delta = radius_km * 1000.0 / max(1.0, meters_per_lon_degree)
    limit = max(1, min(int(limit), 200))
    query_box = {
        "south": max(-90.0, latitude - lat_delta),
        "west": max(-180.0, longitude - lon_delta),
        "north": min(90.0, latitude + lat_delta),
        "east": min(180.0, longitude + lon_delta),
    }
    event_type_normalized = str(event_type or "").strip().lower()
    matches: list[dict[str, Any]] = []
    total_matches = 0
    matched_event_ids: set[str] = set()
    year_counts: dict[int, int] = {}
    for record in _catalog_vil_records():
        if year is not None and record["year"] != int(year):
            continue
        if month is not None and record["month"] != int(month):
            continue
        if event_type_normalized and event_type_normalized not in record["event_type"].lower():
            continue
        south, west = record["llcrnrlat"], record["llcrnrlon"]
        north, east = record["urcrnrlat"], record["urcrnrlon"]
        if None in (south, west, north, east):
            continue
        # Use the closest point of the catalog footprint to the query point
        # and evaluate that distance geodesically.  The latitude/longitude
        # envelope remains in the response for transparency, but it is no
        # longer the criterion that admits a record.
        distance_km = geodesic_distance_to_bounds_km(
            latitude, longitude,
            {"south": south, "west": west, "north": north, "east": east},
        )
        if distance_km > radius_km:
            continue
        total_matches += 1
        matched_event_ids.add(str(record["id"]))
        year_counts[record["year"]] = year_counts.get(record["year"], 0) + 1
        if len(matches) < limit:
            item = _catalog_public_record(record)
            item["bounding_box"] = {
                "south": south, "west": west, "north": north, "east": east,
            }
            item["year"] = record["year"]
            item["month"] = record["month"]
            item["spatial_match"] = "catalog_footprint_within_geodesic_radius"
            item["distance_to_footprint_km"] = round(distance_km, 3)
            matches.append(item)
    return {
        "status": "ok",
        "task": "sevir_catalog_region_search",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "query": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "radius_degrees_approx": {"latitude": lat_delta, "longitude": lon_delta},
            "bounding_box": query_box,
            "month": month,
            "year": year,
            "event_type": event_type,
        },
        "total_matches": total_matches,
        "unique_event_count": len(matched_event_ids),
        "coverage_years": sorted(year_counts),
        "year_counts": {str(year): count for year, count in sorted(year_counts.items())},
        "returned_count": len(matches),
        "displayed_count": len(matches),
        "distance_method": "WGS84 geodesic distance to the nearest point of the CATALOG footprint",
        "records": matches,
        "interpretation_limit": (
            "Matched by the shortest WGS84 geodesic distance from the query point to the catalog event bounding box. "
            "This is metadata; retrieve HDF5 observations for pixel-level analysis."
        ),
    }


def load_request_dataset(description: dict[str, Any]) -> dict[str, Any]:
    """Load request-bound SEVIR data using the repository's observation contract."""
    event_id = description["event_id"]
    metadata = get_sevir_event_metadata(event_id)
    if metadata["status"] != "ok":
        raise ValueError(f"SEVIR metadata unavailable for {event_id}")
    record = {
        "source": "sevir", "dataset_id": f"sevir:{event_id}", "event_id": event_id,
        "status": "available", "metadata": metadata["record"], "role": "event_metadata",
        "capabilities": ["event_metadata", "event_time", "event_location"],
    }
    if description.get("representation") != "observation_sequence":
        return record
    loaded = load_sevir_vil_event(event_id)
    if loaded["status"] != "ok":
        raise ValueError(f"SEVIR observations unavailable for {event_id}: {loaded['status']}")
    values = loaded["frames"]
    minutes = loaded["relative_minutes"]
    window = description.get("observation_window_minutes")
    if window is not None:
        lower, upper = window
        positions = [index for index, minute in enumerate(minutes) if lower <= minute <= upper]
        if not positions:
            raise ValueError(f"SEVIR observation window {lower}..{upper} is empty for {event_id}")
        values = values[positions[0]:positions[-1] + 1]
        minutes = minutes[positions[0]:positions[-1] + 1]
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    record.update(
        role="local_observation", dataset_id=f"sevir:{event_id}:observations", kind="sequence",
        capabilities=["encoded_vil", "local_observation", "temporal_sequence", "event_metadata", "event_time", "event_location"],
        shape=list(values.shape), dimensions=loaded["dimensions"], dtype=str(values.dtype),
        source_dtype=str(values.dtype), missing_value_code=loaded["missing_value"],
        relative_minutes=list(minutes), raw_data=values, values=values,
    )
    return record
