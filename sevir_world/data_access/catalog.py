from __future__ import annotations

import base64
import binascii
import csv
import json
import re
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import *

import h5py
import numpy as np
try:
    import fsspec
except ImportError:  # Local-only deployments do not need the optional remote reader.
    fsspec = None  # type: ignore[assignment]
from pyproj import CRS, Transformer, Geod


def _catalog_number(value: Any) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _catalog_integer(value: Any) -> int | None:
    number = _catalog_number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _minute_offsets(value: Any) -> tuple[int, ...]:
    raw = str(value or "").strip()
    if not raw:
        return ()
    try:
        return tuple(int(item) for item in raw.split(":"))
    except ValueError as error:
        raise ValueError(f"Invalid SEVIR minute_offsets: {raw}") from error


def _frame_times(time_utc: str, offsets: tuple[int, ...]) -> list[str]:
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
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


@lru_cache(maxsize=1)
def _catalog_vil_records() -> tuple[dict[str, Any], ...]:
    if not CATALOG_CSV.exists():
        raise RuntimeError("未找到SEVIR CATALOG.csv。")
    records: list[dict[str, Any]] = []
    with CATALOG_CSV.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("img_type", "")).lower() != "vil":
                continue
            time_utc = str(row.get("time_utc", ""))
            records.append(
                {
                    "id": str(row.get("id", "")),
                    "file_name": str(row.get("file_name", "")),
                    "file_index": int(row["file_index"]) if row.get("file_index", "").isdigit() else None,
                    "img_type": "vil",
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
        raise RuntimeError("SEVIR CATALOG中没有找到VIL记录。")
    return tuple(records)


def _catalog_public_record(record: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: record.get(key)
        for key in (
            "id", "time_utc", "event_type", "file_name", "file_index",
            "data_min", "data_max", "pct_missing",
        )
    }
    public["local_hdf5_available"] = (LOCAL_SEVIR_DATA_ROOT / record["file_name"]).exists()
    public["remote_hdf5_available"] = bool(SEVIR_REMOTE_ENABLED)
    return public


@lru_cache(maxsize=1)
def _catalog_vil_index() -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in _catalog_vil_records():
        key = record["id"].lower()
        grouped.setdefault(key, []).append(record)
    return {key: tuple(records) for key, records in grouped.items()}


def _catalog_vil_event(
    event_id: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    target = str(event_id or "").strip().lower()
    if not target:
        raise ValueError("event_id is required.")
    candidates = list(_catalog_vil_index().get(target, ()))
    if file_name:
        normalized_file = str(file_name).replace("\\", "/").lower()
        candidates = [
            record for record in candidates
            if record["file_name"].replace("\\", "/").lower() == normalized_file
        ]
    if file_index is not None:
        candidates = [record for record in candidates if record["file_index"] == int(file_index)]
    if not candidates:
        raise ValueError(f"SEVIR CATALOG does not contain VIL event {event_id}.")
    if len(candidates) > 1:
        choices = [
            {"file_name": item["file_name"], "file_index": item["file_index"]}
            for item in candidates
        ]
        raise ValueError(
            f"SEVIR VIL id {event_id} has {len(candidates)} CATALOG records; "
            f"provide file_name or file_index. Candidates: {choices}"
        )
    return candidates[0]


def _full_catalog_record(record: dict[str, Any]) -> dict[str, Any]:
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
    root = LOCAL_SEVIR_DATA_ROOT.resolve()
    path = (root / record["file_name"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("CATALOG file_name resolves outside the SEVIR data root.") from error
    return path


def _event_cache_path(record: dict[str, Any]) -> Path:
    """Return a stable event cache path without trusting an event ID as a filename."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["id"]))
    return SEVIR_EVENT_CACHE / f"{safe_id}_{int(record['file_index'])}.npz"


def _remote_hdf5_url(record: dict[str, Any]) -> str:
    relative = str(record["file_name"]).replace("\\", "/").lstrip("/")
    return f"{SEVIR_REMOTE_BASE_URL}/{quote(relative, safe='/')}"


def _read_vil_dataset(handle: Any, record: dict[str, Any]) -> np.ndarray:
    if "vil" not in handle:
        raise ValueError("SEVIR HDF5 does not contain dataset 'vil'.")
    dataset = handle["vil"]
    file_index = record["file_index"]
    if file_index is None:
        raise ValueError(f"SEVIR event {record['id']} has no valid file_index.")
    if not 0 <= file_index < dataset.shape[0]:
        raise IndexError(
            f"file_index {file_index} is outside dataset 'vil' with {dataset.shape[0]} events."
        )
    sample = np.asarray(dataset[file_index])
    expected_frames = len(record["minute_offsets"])
    if sample.ndim != 3:
        raise ValueError(f"Expected a 3-D VIL event, got shape {sample.shape}.")
    time_axes = [axis for axis, size in enumerate(sample.shape) if size == expected_frames]
    if expected_frames and len(time_axes) == 1:
        sample = np.moveaxis(sample, time_axes[0], 0)
    elif sample.shape[-1] == 49:
        sample = np.moveaxis(sample, -1, 0)
    elif sample.shape[0] != 49:
        raise ValueError(f"Cannot identify the time axis in VIL shape {sample.shape}.")
    return np.ascontiguousarray(sample, dtype=np.uint8)


def _load_remote_vil_array(record: dict[str, Any]) -> tuple[np.ndarray, str]:
    cache_path = _event_cache_path(record)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_id = str(np.asarray(payload["event_id"]).item())
            if cached_id == str(record["id"]):
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
            source_url=np.asarray(url),
            file_index=np.asarray(record["file_index"]),
            frames=sample,
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return sample, f"remote:{url}"


def _load_vil_array(record: dict[str, Any]) -> tuple[np.ndarray, str]:
    path = _local_hdf5_path(record)
    if path.exists():
        with h5py.File(path, "r") as handle:
            return _read_vil_dataset(handle, record), str(path)
    return _load_remote_vil_array(record)


def load_sevir_vil_event(
    event_id: str,
    start_frame: int = 0,
    end_frame: int = 49,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    record = _catalog_vil_event(event_id, file_name=file_name, file_index=file_index)
    try:
        cache_path = ROOT / "data" / "live_cascast_cache" / f"{record['id']}_sevir_vil_49frames.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                frames = np.asarray(cached["frames"])
            if frames.ndim == 3 and frames.shape[-1] == len(record["minute_offsets"]):
                frames = np.moveaxis(frames, -1, 0)
            source = str(cache_path)
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


def load_sevir_vil_frame_histograms(
    event_ids: list[str],
    start_frame: int = 0,
    end_frame: int = 49,
) -> dict[str, Any]:
    """Load bounded local HDF5 events as lossless per-frame value histograms.

    This is a data primitive, not a duration report: it chooses no VIL
    threshold, active-area criterion, similarity rule, or summary statistic.
    Generated Agent code can select those quantities and derive durations from
    the returned encoded-value counts and minute offsets without transporting
    many full 384x384 arrays through the JSON boundary.
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


def search_sevir_catalog(
    query: str = "",
    year: int | None = None,
    month: int | None = None,
    event_type: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    query_normalized = str(query or "").strip().lower()
    event_type_normalized = str(event_type or "").strip().lower()
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
        total_matches += 1
        if len(matches) < limit:
            matches.append(_catalog_public_record(record))
    return {
        "status": "ok",
        "task": "sevir_catalog_search",
        "data_source": str(CATALOG_CSV.relative_to(ROOT)),
        "channel": "vil",
        "filters": {"query": query, "year": year, "month": month, "event_type": event_type},
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
    """Find VIL events whose CATALOG footprint intersects a point window.

    This is deliberately a metadata-only operation.  It lets the Agent select
    historical candidates before any remote HDF5 bytes are fetched.
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
