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

from .config import *

import h5py
import numpy as np
from pyproj import CRS, Transformer


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


def _load_vil_array(record: dict[str, Any]) -> tuple[np.ndarray, Path]:
    path = _local_hdf5_path(record)
    if not path.exists():
        raise FileNotFoundError(
            f"The indexed HDF5 file is not available locally: {record['file_name']}"
        )
    file_index = record["file_index"]
    if file_index is None:
        raise ValueError(f"SEVIR event {record['id']} has no valid file_index.")
    with h5py.File(path, "r") as handle:
        if "vil" not in handle:
            raise ValueError(f"HDF5 file {path.name} does not contain dataset 'vil'.")
        dataset = handle["vil"]
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
    return sample, path


def load_sevir_vil_event(
    event_id: str,
    start_frame: int = 0,
    end_frame: int = 49,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    record = _catalog_vil_event(event_id, file_name=file_name, file_index=file_index)
    indexed_path = _local_hdf5_path(record)
    if not indexed_path.exists():
        return {
            "status": "data_unavailable",
            "task": "load_sevir_vil_event",
            "event_id": record["id"],
            "pixel_data_loaded": False,
            "required_hdf5": record["file_name"],
            "file_index": record["file_index"],
            "metadata_available": True,
            "frame_times_utc": _frame_times(record["time_utc"], record["minute_offsets"]),
            "reason": "CATALOG metadata is available, but the indexed official HDF5 file is not stored locally.",
        }
    frames, path = _load_vil_array(record)
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
        "source_hdf5": str(path.relative_to(LOCAL_SEVIR_DATA_ROOT.resolve())),
        "file_index": record["file_index"],
        "full_event_shape": [int(value) for value in frames.shape],
        "loaded_shape": [int(value) for value in selected.shape],
        "frame_interval": {"start_inclusive": start, "end_exclusive": end},
        "frame_statistics": frame_statistics,
        "pixel_data_loaded": True,
        "pixel_data_returned_to_llm": False,
        "interpretation_limit": (
            "The HDF5 pixels were read for computation, but the full array is not embedded in the "
            "tool response. VIL value 255 is treated as missing."
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
