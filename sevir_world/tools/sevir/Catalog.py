from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import json
import re
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ...data_access.config import *

import h5py
import numpy as np
try:
    import fsspec
except ImportError:  # Local-only deployments do not need the optional remote reader.
    fsspec = None  # type: ignore[assignment]
from pyproj import CRS, Transformer, Geod


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
    try:
        reference = datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return []
    return [
        (reference + timedelta(minutes=offset)).isoformat().replace("+00:00", "Z")
        for offset in offsets
    ]


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
        raise RuntimeError("未找到SEVIR CATALOG.csv。")
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
        raise RuntimeError("SEVIR CATALOG中没有找到记录。")
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
    public["observation_cache_available"] = (
        ROOT / "data" / "live_cascast_cache" /
        f"{record['id']}_sevir_vil_49frames.npz"
    ).is_file()
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
        if record is None or record.get("img_type") != img_type:
            raise ValueError(f"SEVIR CATALOG does not contain {img_type} subject {event_id}.")
        return record

    # Build index on-demand for this img_type
    candidates = [r for r in _catalog_all_records() if r["id"].lower() == target and r["img_type"] == img_type]

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
    return SEVIR_EVENT_CACHE / f"{safe_id}_{int(record['file_index'])}.npz"


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


def _load_remote_array_by_type(record: dict[str, Any]) -> tuple[np.ndarray, str]:
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
    cache_path = _event_cache_path(record)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_id = str(np.asarray(payload["event_id"]).item())
            cached_index = (
                int(np.asarray(payload["file_index"]).item())
                if "file_index" in payload.files else None
            )
            cached_type = str(np.asarray(payload.get("img_type", "vil")).item())
            if (cached_id == str(record["id"])
                and cached_index == record["file_index"]
                and cached_type == record["img_type"]):
                return np.asarray(payload["frames"]), f"cache:{cache_path}"

    if not SEVIR_REMOTE_ENABLED:
        raise FileNotFoundError(
            f"The HDF5 file for {record['id']} ({record['img_type']}) is not local and remote SEVIR access is disabled."
        )
    if fsspec is None:
        raise RuntimeError(
            "Remote SEVIR access requires the optional 'fsspec' package; "
            "install fsspec or set SEVIR_REMOTE_ENABLED=0."
        )

    url = _remote_hdf5_url(record)
    try:
        with fsspec.open(
            url,
            mode="rb",
            block_size=SEVIR_REMOTE_BLOCK_SIZE,
            cache_type="blockcache",
        ) as remote_file:
            with h5py.File(remote_file, "r") as handle:
                sample = _read_dataset_by_type(handle, record)
    except Exception as error:
        raise RuntimeError(
            f"Unable to read remote SEVIR HDF5 for {record['id']} ({record['img_type']}) from {url}: {error}"
        ) from error

    SEVIR_EVENT_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            event_id=np.asarray(record["id"]),
            img_type=np.asarray(record["img_type"]),
            subject_key=np.asarray(_catalog_subject_key(record)),
            source_url=np.asarray(url),
            file_index=np.asarray(record["file_index"]),
            frames=sample,
        )
        temporary.replace(cache_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sample, url


def _load_remote_vil_array(record: dict[str, Any]) -> tuple[np.ndarray, str]:
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
    cache_path = _event_cache_path(record)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_id = str(np.asarray(payload["event_id"]).item())
            cached_index = (
                int(np.asarray(payload["file_index"]).item())
                if "file_index" in payload.files else None
            )
            if cached_id == str(record["id"]) and cached_index == record["file_index"]:
                return np.asarray(payload["frames"], dtype=np.uint8), f"cache:{cache_path}"

    if not SEVIR_REMOTE_ENABLED:
        raise FileNotFoundError(
            f"The HDF5 file for {record['id']} is not local and remote SEVIR access is disabled."
        )
    if fsspec is None:
        raise RuntimeError(
            "Remote SEVIR access requires the optional 'fsspec' package; "
            "install fsspec or set SEVIR_REMOTE_ENABLED=0."
        )

    url = _remote_hdf5_url(record)
    try:
        with fsspec.open(
            url,
            mode="rb",
            block_size=SEVIR_REMOTE_BLOCK_SIZE,
            cache_type="blockcache",
        ) as remote_file:
            with h5py.File(remote_file, "r") as handle:
                sample = _read_vil_dataset(handle, record)
    except Exception as error:
        raise RuntimeError(
            f"Unable to read remote SEVIR HDF5 for {record['id']} from {url}: {error}"
        ) from error

    SEVIR_EVENT_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            event_id=np.asarray(record["id"]),
            subject_key=np.asarray(_catalog_subject_key(record)),
            source_url=np.asarray(url),
            file_index=np.asarray(record["file_index"]),
            frames=sample,
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return sample, f"remote:{url}"


def _load_vil_array(record: dict[str, Any]) -> tuple[np.ndarray, str]:
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
    return _load_array_by_type({**record, "img_type": "vil"})


def _load_array_by_type(record: dict[str, Any]) -> tuple[np.ndarray, str]:
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
    local_path = _local_hdf5_path(record)
    if local_path.is_file():
        with h5py.File(local_path, "r") as handle:
            return _read_dataset_by_type(handle, record), str(local_path)
    return _load_remote_array_by_type(record)


def load_sevir_vil_event(
    event_id: str,
    start_frame: int = 0,
    end_frame: int = 49,
    file_name: str = "",
    file_index: int | None = None,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
) -> dict[str, Any]:
    """Load a bounded frame interval for one SEVIR VIL event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    start_frame : int, default 0
        Inclusive zero-based first frame.
    end_frame : int, default 49
        Inclusive zero-based last frame.
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

        cube = load_sevir_vil_event(event_id, 0, 12)
    """
    record = _catalog_vil_event(event_id, file_name=file_name, file_index=file_index)
    try:
        legacy_cache_path = (
            ROOT / "data" / "live_cascast_cache" /
            f"{record['id']}_sevir_vil_49frames.npz"
        )
        # Old demonstration caches were named only by event_id and contain no
        # catalog-row discriminator. They are safe to reuse only while that ID
        # resolves to exactly one VIL row. Duplicate IDs must use the canonical
        # subject-key/file-index cache path instead of guessing.
        unique_human_id = len(_catalog_vil_index().get(str(record["id"]).lower(), ())) == 1
        if legacy_cache_path.is_file() and unique_human_id:
            with np.load(legacy_cache_path, allow_pickle=False) as cached:
                frames = np.asarray(cached["frames"])
            if frames.ndim == 3 and frames.shape[-1] == len(record["minute_offsets"]):
                frames = np.moveaxis(frames, -1, 0)
            source = str(legacy_cache_path)
        else:
            frames, source = _load_vil_array(record)
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "status": "data_unavailable",
            "task": "load_sevir_vil_event",
            "event_id": record["id"],
            "pixel_data_loaded": False,
            "required_hdf5": record["file_name"],
            "file_index": record["file_index"],
            "metadata_available": True,
            "frame_times_utc": _frame_times(record["time_utc"], record["minute_offsets"]),
            "reason": str(error),
        }
    if minutes_before is not None or minutes_after is not None:
        before = max(0, int(minutes_before or 0))
        after = max(0, int(minutes_after or 0))
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
    frame_statistics = []
    for position, frame in enumerate(selected, start=start):
        valid = frame[frame != 255]
        frame_statistics.append(
            {
                "frame_index": position,
                "minute_offset": record["minute_offsets"][position] if position < len(record["minute_offsets"]) else None,
                "time_utc": times[position - start] if position - start < len(times) else None,
                "min_vil": int(valid.min()) if valid.size else None,
                "max_vil": int(valid.max()) if valid.size else None,
                "mean_vil": round(float(valid.mean()), 4) if valid.size else None,
                "missing_pixel_count": int((frame == 255).sum()),
            }
        )
    return {
        "status": "ok",
        "task": "load_sevir_vil_event",
        "event_id": record["id"],
        "subject_key": _catalog_subject_key(record),
        "event_anchor_time_utc": record["time_utc"],
        "frame_times_utc": times,
        "source_hdf5": source,
        "file_index": record["file_index"],
        "full_event_shape": [int(value) for value in frames.shape],
        "loaded_shape": [int(value) for value in selected.shape],
        "frame_interval": {"start_inclusive": start, "end_exclusive": end},
        "frame_statistics": frame_statistics,
        # Full frames stay inside the isolated generated-code worker as a
        # read-only ndarray.  The worker/trace serializers replace large
        # arrays with shape metadata, so pixels are available for Agent-written
        # calculations without being copied into the LLM observation.
        "frames": np.ascontiguousarray(selected, dtype=np.float32),
        "values": np.ascontiguousarray(selected, dtype=np.float32),
        "relative_minutes": list(offsets),
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


def load_sevir_vil_frame_histograms(
    event_ids: list[str],
    start_frame: int = 0,
    end_frame: int = 49,
) -> dict[str, Any]:
    """Load bounded events as lossless per-frame value histograms.

    Parameters
    ----------
    event_ids : list of str
        Indexed SEVIR event identifiers.
    start_frame, end_frame : int
        Inclusive zero-based frame interval.
    max_events : int
        Maximum events loaded in one call.

    Returns
    -------
    dict
        Encoded-value counts, minute offsets and event provenance. No
        threshold, duration rule or statistic is selected by this function.
    """
    identifiers = [str(value).strip() for value in event_ids if str(value).strip()]
    identifiers = list(dict.fromkeys(identifiers))
    if not identifiers:
        raise ValueError("event_ids must contain at least one SEVIR event ID.")
    if len(identifiers) > 10:
        raise ValueError("event_ids is limited to 10 events per HDF5 histogram call.")
    start = max(0, int(start_frame))
    end = min(49, int(end_frame))
    if start >= end:
        raise ValueError(f"Invalid frame interval [{start_frame}, {end_frame}).")

    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for event_id in identifiers:
        try:
            record = _catalog_vil_event(event_id)
            frames, source = _load_vil_array(record)
        except (ValueError, FileNotFoundError, IndexError, RuntimeError) as error:
            unavailable.append({"event_id": event_id, "reason": str(error)})
            continue
        selected = frames[start:min(end, int(frames.shape[0]))]
        offsets = record["minute_offsets"][start:start + int(selected.shape[0])]
        histograms = [
            np.bincount(np.asarray(frame, dtype=np.uint8).reshape(-1), minlength=256)
            .astype(np.int64)
            .tolist()
            for frame in selected
        ]
        records.append({
            "event_id": record["id"],
            "event_type": record["event_type"],
            "time_utc": record["time_utc"],
            "catalog_data_max": record["data_max"],
            "source_hdf5": source,
            "file_index": record["file_index"],
            "minute_offsets": list(offsets),
            "frame_histograms": histograms,
        })
    return {
        "status": "ok" if records else "data_unavailable",
        "task": "sevir_vil_frame_histograms",
        "requested_event_count": len(identifiers),
        "loaded_event_count": len(records),
        "unavailable_event_count": len(unavailable),
        "value_bins": list(range(256)),
        "missing_value": 255,
        "records": records,
        "unavailable": unavailable,
        "capabilities": ["historical_observation", "hdf5_pixels", "agent_defined_duration"],
        "interpretation_limit": (
            "Each frame_histograms row is an exact count for encoded VIL values 0..255. "
            "No threshold, active-frame rule, duration, similarity rule, or aggregate was "
            "preselected. Bin 255 is missing and must be excluded."
        ),
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
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc))


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
        return not (
            float(left["east"]) < float(right["west"])
            or float(left["west"]) > float(right["east"])
            or float(left["north"]) < float(right["south"])
            or float(left["south"]) > float(right["north"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def search_sevir_catalog(
    query: str = "",
    year: int | None = None,
    month: int | None = None,
    event_type: str = "",
    region: dict[str, float] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius_km: float | None = None,
    start_time_utc: str | None = None,
    end_time_utc: str | None = None,
    require_prior_minutes: int | None = None,
    require_observation_cache: bool = False,
    limit: int = 10,
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
    region = dict(region) if isinstance(region, dict) else None
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
    start_dt = _parse_query_time(start_time_utc)
    end_dt = _parse_query_time(end_time_utc)
    prior = int(require_prior_minutes) if require_prior_minutes is not None else None
    limit = max(1, min(int(limit), 50))
    matches: list[dict[str, Any]] = []
    total_matches = 0
    for record in _catalog_vil_records():
        searchable = " ".join(
            str(record.get(key, "")) for key in ("id", "file_name", "time_utc", "event_type")
        ).lower()
        if query_normalized and query_normalized not in searchable:
            continue
        if year is not None and record["year"] != int(year):
            continue
        if month is not None and record["month"] != int(month):
            continue
        if event_type_normalized and event_type_normalized not in record["event_type"].lower():
            continue
        record_dt = _parse_query_time(record["time_utc"])
        if start_dt and (record_dt is None or record_dt < start_dt):
            continue
        if end_dt and (record_dt is None or record_dt > end_dt):
            continue
        if prior is not None:
            available_offsets = {int(offset) for offset in record["minute_offsets"]}
            required_offsets = set(range(-abs(prior), 1, 5))
            if not required_offsets.issubset(available_offsets):
                continue
        if require_observation_cache and not (
            ROOT / "data" / "live_cascast_cache" /
            f"{record['id']}_sevir_vil_49frames.npz"
        ).is_file():
            continue
        footprint = {
            "south": record["llcrnrlat"], "west": record["llcrnrlon"],
            "north": record["urcrnrlat"], "east": record["urcrnrlon"],
        }
        if region and not _boxes_intersect(region, footprint):
            continue
        distance_km: float | None = None
        if point is not None and geod is not None and distance_limit_km is not None:
            if any(footprint[key] is None for key in footprint):
                continue
            closest_lat = min(max(point[0], float(footprint["south"])), float(footprint["north"]))
            closest_lon = min(max(point[1], float(footprint["west"])), float(footprint["east"]))
            _, _, distance_m = geod.inv(point[1], point[0], closest_lon, closest_lat)
            distance_km = float(distance_m) / 1000.0
            if distance_km > distance_limit_km:
                continue
        total_matches += 1
        if len(matches) < limit:
            item = _catalog_public_record(record)
            offsets = sorted({int(value) for value in record["minute_offsets"]})
            required_offsets = list(range(-abs(prior), 1, 5)) if prior is not None else []
            item.update({
                "history_span_minutes": max(0, -min(offsets)) if offsets else 0,
                "history_expected_frames": len(required_offsets),
                "history_available_frames": sum(value in offsets for value in required_offsets),
                "history_is_continuous": (
                    all(value in offsets for value in required_offsets)
                    if required_offsets else None
                ),
            })
            if distance_km is not None:
                item["distance_to_footprint_km"] = round(distance_km, 3)
            matches.append(item)
    return {
        "status": "ok",
        "task": "sevir_catalog_search",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "filters": {"query": query, "year": year, "month": month, "event_type": event_type,
                    "region": region, "latitude": latitude, "longitude": longitude,
                    "radius_km": distance_limit_km, "start_time_utc": start_time_utc,
                    "end_time_utc": end_time_utc, "require_prior_minutes": prior,
                    "require_observation_cache": bool(require_observation_cache)},
        "total_matches": total_matches,
        "returned_count": len(matches),
        "records": matches,
        "interpretation_limit": "结果来自SEVIR CATALOG metadata，不代表已下载对应HDF5像素数据。",
    }


def search_sevir_catalog_by_region(
    latitude: float,
    longitude: float,
    radius_degrees: float | None = None,
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
    radius_degrees : float
        Half-width of the latitude/longitude window.
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
    if radius_degrees is not None:
        # Backward-compatible transport for old callers. New callers should
        # use radius_km so the user-facing distance has a physical meaning.
        radius_km = max(0.0, float(radius_degrees) * 111.32)
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
        closest_lat = min(max(latitude, float(south)), float(north))
        closest_lon = min(max(longitude, float(west)), float(east))
        _, _, distance_m = geod.inv(longitude, latitude, closest_lon, closest_lat)
        if float(distance_m) > radius_km * 1000.0:
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
            item["distance_to_footprint_km"] = round(float(distance_m) / 1000.0, 3)
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
        "statistics_count": len(matched_event_ids),
        "distance_method": "WGS84 geodesic distance to the nearest point of the CATALOG footprint",
        "records": matches,
        "interpretation_limit": (
            "匹配依据是CATALOG事件包围盒到查询点的WGS84测地线最近距离，不代表事件核心位于该地点；"
            "返回的是元数据，像素仍需按需读取HDF5。"
        ),
    }
