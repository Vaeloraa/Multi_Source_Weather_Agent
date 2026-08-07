from __future__ import annotations

import json
import os
import sys
import base64
import binascii
import csv
import threading
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402
import h5py  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402


SAMPLE_DIR = ROOT / "data" / "sevir_sample"
SAMPLE_NPZ = SAMPLE_DIR / "sevir_vil_event.npz"
SAMPLE_METADATA = SAMPLE_DIR / "metadata.json"
CASCAST_ROOT = Path(os.getenv("CASCAST_ROOT", str(ROOT.parent / "CasCast")))
EARTHFORMER_LATENT_ROOT = Path(
    os.getenv("EARTHFORMER_LATENT_ROOT", str(ROOT / "earthformer_example"))
)
CASCAST_EVENT_ID = "R19100105107994"
CASCAST_LEGACY_EVENT_ID = "R19082500047459"
CASCAST_SAMPLE_NAME = "vil-2019-SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5-0-0.npy"
CASCAST_OUTPUT_NPZ = (
    CASCAST_ROOT
    / "experiments"
    / "cascast_diffusion"
    / "world_size1-ckpt"
    / "local_ens1_outputs"
    / CASCAST_SAMPLE_NAME.replace(".npy", ".npz")
)
CASCAST_METADATA = ROOT / "data" / "cascast_sample" / "metadata.json"
CASCAST_ENSEMBLE_CACHE = ROOT / "data" / "cascast_ensemble_cache"
CASCAST_ENSEMBLE_MANIFEST = CASCAST_ENSEMBLE_CACHE / "manifest.json"
UPLOADED_VIL: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOADED_VIL_SEQUENCES: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
UPLOAD_STORAGE_DIR = ROOT / "data" / "uploads"
UPLOAD_STORAGE_LOCK = threading.Lock()
CATALOG_CSV = ROOT / "data" / "sevir_complete" / "CATALOG.csv"
LOCAL_SEVIR_DATA_ROOT = ROOT / "data" / "sevir_complete" / "data"
CASCAST_EVALUATION_REPORT = ROOT / "data" / "cascast_evaluation_report.json"
SEVIR_EVALUATION_THRESHOLDS = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)


def _earthformer_latent_status() -> dict[str, Any]:
    files = sorted(EARTHFORMER_LATENT_ROOT.rglob("*.npy")) if EARTHFORMER_LATENT_ROOT.exists() else []
    valid_files: list[str] = []
    invalid_files: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = [int(value) for value in payload.shape]
            if payload.ndim == 4 and shape[0] == 12 and shape[1:] == [4, 48, 48]:
                valid_files.append(str(path.relative_to(ROOT)))
            else:
                invalid_files.append({"file": str(path), "shape": shape})
        except (OSError, ValueError) as error:
            invalid_files.append({"file": str(path), "error": str(error)})
    matching_latent = any(Path(path).name == CASCAST_SAMPLE_NAME for path in valid_files)
    return {
        "available": bool(valid_files),
        "root": str(EARTHFORMER_LATENT_ROOT),
        "valid_file_count": len(valid_files),
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "expected_shape": [12, 4, 48, 48],
        "event_compatible": matching_latent,
        "event_id": CASCAST_LEGACY_EVENT_ID if matching_latent else None,
        "event_note": (
            f"已匹配旧版单事件{CASCAST_LEGACY_EVENT_ID}。"
            if matching_latent
            else f"未找到旧版单事件{CASCAST_LEGACY_EVENT_ID}对应的EarthFormer latent。"
        ),
    }


@lru_cache(maxsize=1)
def _ensemble_manifest() -> dict[str, Any]:
    if not CASCAST_ENSEMBLE_MANIFEST.exists():
        raise RuntimeError("未找到CasCast多事件ensemble缓存，请先运行build_ensemble_cache.py。")
    return json.loads(CASCAST_ENSEMBLE_MANIFEST.read_text(encoding="utf-8"))


def _ensemble_event(event_id: str) -> dict[str, Any]:
    manifest = _ensemble_manifest()
    requested = str(event_id).strip()
    if requested in {"sample", "event_001", ""}:
        requested = CASCAST_EVENT_ID
    for event in manifest["events"]:
        aliases = {
            str(event["event_id"]),
            str(event["sample_index"]),
            f"SEVIR_TEST_{event['sample_index']}",
        }
        if requested in aliases:
            return event
    available = "、".join(str(event["event_id"]) for event in manifest["events"])
    raise ValueError(f"当前CasCast结果不含事件{requested}；可选事件：{available}")


@lru_cache(maxsize=3)
def _load_cascast_ensemble(event_id: str = CASCAST_EVENT_ID) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    event = _ensemble_event(event_id)
    cache_path = CASCAST_ENSEMBLE_CACHE / str(event["cache_file"])
    if not cache_path.exists():
        raise RuntimeError(f"CasCast事件缓存不存在：{cache_path.name}")
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {
            "history": np.asarray(payload["history"], dtype=np.uint8),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.uint8),
            "ensemble_members": np.asarray(payload["ensemble_members"], dtype=np.float32),
            "ensemble_mean": np.asarray(payload["ensemble_mean"], dtype=np.float32),
            "pixel_std": np.asarray(payload["pixel_std"], dtype=np.float32),
            "q10": np.asarray(payload["q10"], dtype=np.float32),
            "q90": np.asarray(payload["q90"], dtype=np.float32),
            "member_seed": np.asarray(payload["member_seed"], dtype=np.int64),
        }
    if arrays["ensemble_members"].shape != (10, 12, 384, 384):
        raise RuntimeError(f"CasCast ensemble形状不正确：{arrays['ensemble_members'].shape}")
    metadata = {
        **event,
        "event_id": event["event_id"],
        "ensemble_count": int(arrays["ensemble_members"].shape[0]),
        "ground_truth_available": True,
        "catalog_row": {
            "llcrnrlat": event.get("llcrnrlat"),
            "llcrnrlon": event.get("llcrnrlon"),
            "urcrnrlat": event.get("urcrnrlat"),
            "urcrnrlon": event.get("urcrnrlon"),
        },
    }
    return arrays, metadata


def _load_cascast_prediction(event_id: str = CASCAST_EVENT_ID) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, metadata = _load_cascast_ensemble(event_id)
    prediction = arrays["ensemble_mean"].transpose(1, 2, 0)
    return prediction, metadata


def _vil_display_values(values: np.ndarray) -> np.ndarray:
    """Return a valid SEVIR display/engineering view without mutating raw output."""
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 254.0)


def _cascast_source(event_id: str = CASCAST_EVENT_ID) -> dict[str, Any]:
    event = _ensemble_event(event_id)
    return {
        "kind": "cascast_official_test_prediction",
        "event_id": event["event_id"],
        "sample_name": event["sample_name"],
        "prediction_file": str((CASCAST_ENSEMBLE_CACHE / event["cache_file"]).relative_to(ROOT)),
        "metadata_file": str(CASCAST_ENSEMBLE_MANIFEST.relative_to(ROOT)),
        "note": "CasCast真实推理结果：13帧历史观测、10路ensemble、ensemble mean与12帧ground truth。",
    }


def _source() -> dict[str, Any]:
    return {
        "kind": "official_sevir_sample",
        "sample_file": str(SAMPLE_NPZ.relative_to(ROOT)),
        "metadata_file": str(SAMPLE_METADATA.relative_to(ROOT)),
        "note": "由官方SEVIR VIL文件提取的单个事件；不是完整SEVIR数据集。",
    }


@lru_cache(maxsize=1)
def _load_sample() -> tuple[np.ndarray, dict[str, Any]]:
    if not SAMPLE_NPZ.exists() or not SAMPLE_METADATA.exists():
        raise RuntimeError("未找到本地SEVIR样本，请先运行scripts/fetch_sevir_sample.py。")
    metadata = json.loads(SAMPLE_METADATA.read_text(encoding="utf-8"))
    with np.load(SAMPLE_NPZ, allow_pickle=False) as payload:
        vil = np.asarray(payload["vil"], dtype=np.uint8)
        event_id = str(payload["event_id"].item())
    if vil.ndim != 3 or vil.shape[-1] < 2:
        raise RuntimeError(f"SEVIR样本形状不正确：{vil.shape}")
    metadata["event_id"] = event_id
    return vil, metadata


def _event(event_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    vil, metadata = _load_sample()
    actual_id = metadata["event_id"]
    if event_id not in {actual_id, "event_001", "sample"}:
        raise ValueError(f"本地仅有事件{actual_id}，没有找到{event_id}。")
    return vil, metadata


def _preview(frame: np.ndarray) -> list[list[int]]:
    height, width = frame.shape
    target = 12
    if height % target == 0 and width % target == 0:
        pooled = frame.reshape(target, height // target, target, width // target).max(
            axis=(1, 3)
        )
    else:
        y_index = np.linspace(0, height - 1, target).astype(int)
        x_index = np.linspace(0, width - 1, target).astype(int)
        pooled = frame[np.ix_(y_index, x_index)]
    return pooled.astype(int).tolist()


def get_cascast_display_frame(
    event_id: str,
    series: str,
    frame_index: int = 0,
    model_name: str = "cascast",
    threshold: float = 133,
    lead_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one full-resolution frame for the web viewer.

    The 12x12 preview grids remain API summaries only. This function keeps the
    displayed history and forecast on the original 384x384 grid.
    """
    arrays, metadata = _load_cascast_ensemble(event_id)
    history = arrays["history"].astype(np.float32)
    if model_name == "cascast":
        forecast = arrays["ensemble_mean"].astype(np.float32)
    elif model_name == "persistence_baseline":
        forecast = np.repeat(history[-1][None, :, :], 12, axis=0)
    else:
        raise ValueError(f"不支持的预测模型：{model_name}")

    if series == "history":
        frames = history
    elif series == "forecast":
        frames = forecast
    elif series == "forecast_overview":
        return forecast.max(axis=0), metadata
    elif series == "ground_truth":
        truth = arrays["ground_truth"]
        if not 0 <= frame_index < truth.shape[0]:
            raise ValueError(f"Ground truth帧序号超出范围：{frame_index}")
        return truth[frame_index].astype(np.float32), metadata
    elif series == "uncertainty_mean":
        return arrays["ensemble_mean"][-1].astype(np.float32), metadata
    elif series == "uncertainty_std":
        return arrays["pixel_std"].max(axis=0).astype(np.float32), metadata
    elif series == "uncertainty_probability":
        probability = (
            arrays["ensemble_members"].astype(np.float32) >= threshold
        ).mean(axis=0)
        return probability.max(axis=0).astype(np.float32), metadata
    elif series == "hazard_probability_any":
        probability = (arrays["ensemble_members"].astype(np.float32) >= threshold).any(axis=1).mean(axis=0)
        return probability.astype(np.float32), metadata
    elif series == "hazard_duration_mean":
        duration = (arrays["ensemble_members"].astype(np.float32) >= threshold).sum(axis=1).mean(axis=0) * 5.0
        return duration.astype(np.float32), metadata
    elif series in {"ensemble_member_final", "ensemble_member"}:
        members = arrays["ensemble_members"]
        if not 0 <= frame_index < members.shape[0]:
            raise ValueError(f"Ensemble成员序号超出范围：{frame_index}")
        selected_lead = -1 if lead_index is None else int(lead_index)
        if not 0 <= selected_lead < members.shape[1]:
            raise ValueError(f"Ensemble时效序号超出范围：{selected_lead}")
        return members[frame_index, selected_lead].astype(np.float32), metadata
    else:
        raise ValueError("不支持的CasCast图像序列。")
    if not 0 <= frame_index < frames.shape[0]:
        raise ValueError(f"帧序号超出范围：{frame_index}")
    return frames[frame_index], metadata


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


def _pixel_latlon(record: dict[str, Any], row: int, column: int) -> dict[str, Any]:
    size_x = record["size_x"]
    size_y = record["size_y"]
    width_m = record["width_m"]
    height_m = record["height_m"]
    proj = record["proj"]
    if not all((size_x, size_y, width_m, height_m, proj)):
        raise ValueError(f"SEVIR event {record['id']} has incomplete projection metadata.")
    if not 0 <= row < size_y or not 0 <= column < size_x:
        raise ValueError(
            f"Pixel (row={row}, column={column}) is outside image shape ({size_y}, {size_x})."
        )

    projected_crs = CRS.from_proj4(proj)
    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_latlon = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    x_min, _ = to_projected.transform(record["llcrnrlon"], record["llcrnrlat"])
    _, y_max = to_projected.transform(record["urcrnrlon"], record["urcrnrlat"])
    pixel_width_m = width_m / size_x
    pixel_height_m = height_m / size_y
    x = x_min + (column + 0.5) * pixel_width_m
    y = y_max - (row + 0.5) * pixel_height_m
    longitude, latitude = to_latlon.transform(x, y)
    return {
        "row": row,
        "column": column,
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "projected_x_m": round(float(x), 3),
        "projected_y_m": round(float(y), 3),
        "pixel_resolution_m": [round(float(pixel_height_m), 3), round(float(pixel_width_m), 3)],
    }


def geolocate_sevir_event(
    event_id: str,
    row: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    record = _catalog_vil_event(event_id)
    if (row is None) != (column is None):
        raise ValueError("row and column must be provided together.")
    coordinates = (record["llcrnrlat"], record["llcrnrlon"], record["urcrnrlat"], record["urcrnrlon"])
    if any(value is None for value in coordinates):
        raise ValueError(f"SEVIR event {event_id} has incomplete geographic bounds.")
    min_lat, min_lon, max_lat, max_lon = coordinates
    result = {
        "status": "ok",
        "task": "sevir_event_geolocation",
        "event_id": record["id"],
        "time_utc": record["time_utc"],
        "event_type": record["event_type"],
        "bounding_box": {
            "south": min_lat,
            "west": min_lon,
            "north": max_lat,
            "east": max_lon,
        },
        "approx_center_latlon": [round((min_lat + max_lat) / 2, 6), round((min_lon + max_lon) / 2, 6)],
        "projection": record["proj"],
        "image_shape": [record["size_y"], record["size_x"]],
        "physical_extent_m": [record["height_m"], record["width_m"]],
        "file_name": record["file_name"],
        "file_index": record["file_index"],
        "metadata_source": "SEVIR CATALOG",
        "interpretation_limit": (
            "The bounding box and projection come from SEVIR CATALOG. Pixel coordinates are "
            "converted with the supplied projection; no administrative reverse geocoding is performed."
        ),
    }
    if row is not None and column is not None:
        result["pixel_location"] = _pixel_latlon(record, int(row), int(column))
    return result


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


def compare_event_with_history(event_id: str) -> dict[str, Any]:
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
    return {
        "status": "ok",
        "task": "event_history_comparison",
        "event_id": record["id"],
        "event_type": record["event_type"],
        "time_utc": record["time_utc"],
        "catalog_data_max": record["data_max"],
        "reference_group": record["event_type"] if record["event_type"] != "Unlabeled" else "all_vil_records",
        "reference_count": int(values.size),
        "data_max_percentile": round(percentile, 2),
        "interpretation": f"该记录的catalog data_max高于或等于约{percentile:.2f}%的参考记录。",
        "interpretation_limit": "这是catalog data_max的经验分位比较，不是灾害概率或模型预测概率。",
    }


def _valid_upload_id(upload_id: str) -> bool:
    return len(upload_id) == 32 and all(
        character in "0123456789abcdef" for character in upload_id
    )


def _upload_storage_paths(upload_id: str) -> tuple[Path, Path]:
    if not _valid_upload_id(upload_id):
        raise ValueError("Invalid upload_id format.")
    return (
        UPLOAD_STORAGE_DIR / f"{upload_id}.npz",
        UPLOAD_STORAGE_DIR / f"{upload_id}.json",
    )


def _persist_uploaded_array(
    upload_id: str,
    pixels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    array_path, metadata_path = _upload_storage_paths(upload_id)
    UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary_array = UPLOAD_STORAGE_DIR / f".{upload_id}.{nonce}.npz"
    temporary_metadata = UPLOAD_STORAGE_DIR / f".{upload_id}.{nonce}.json"
    stored_metadata = {**metadata, "persisted": True}
    with UPLOAD_STORAGE_LOCK:
        try:
            np.savez_compressed(temporary_array, pixels=np.asarray(pixels, dtype=np.uint8))
            temporary_metadata.write_text(
                json.dumps(stored_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_array.replace(array_path)
            temporary_metadata.replace(metadata_path)
        finally:
            temporary_array.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)


def _restore_uploaded_array(
    upload_id: str,
    expected_kind: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    array_path, metadata_path = _upload_storage_paths(upload_id)
    if not array_path.exists() or not metadata_path.exists():
        return None
    with UPLOAD_STORAGE_LOCK:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("kind") != expected_kind:
            return None
        with np.load(array_path, allow_pickle=False) as payload:
            pixels = np.asarray(payload["pixels"], dtype=np.uint8)
    expected_shape = (384, 384) if expected_kind == "single_frame" else (13, 384, 384)
    if pixels.shape != expected_shape:
        raise ValueError(
            f"Persisted upload {upload_id} has shape {pixels.shape}, expected {expected_shape}."
        )
    return pixels, metadata


def list_uploaded_records() -> dict[str, Any]:
    UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for metadata_path in UPLOAD_STORAGE_DIR.glob("*.json"):
        upload_id = metadata_path.stem
        if not _valid_upload_id(upload_id):
            continue
        array_path, _ = _upload_storage_paths(upload_id)
        if not array_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(
            {
                key: metadata.get(key)
                for key in (
                    "upload_id", "kind", "filename", "filenames", "shape", "dtype",
                    "grayscale_source", "all_grayscale_source", "created_at_utc",
                )
            }
        )
    records.sort(key=lambda item: str(item.get("created_at_utc") or ""), reverse=True)
    return {
        "status": "ok",
        "storage": _path_label(UPLOAD_STORAGE_DIR),
        "record_count": len(records),
        "records": records,
    }


def register_uploaded_vil(
    filename: str,
    width: int,
    height: int,
    pixels_base64: str,
    grayscale_source: bool,
) -> dict[str, Any]:
    if (width, height) != (384, 384):
        raise ValueError("第一版仅支持384×384的SEVIR VIL图片。")
    try:
        raw = base64.b64decode(pixels_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("上传图片的像素数据无法解析。") from error
    if len(raw) != width * height:
        raise ValueError("上传图片的像素数量与384×384尺寸不一致。")
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
    upload_id = uuid.uuid4().hex
    metadata = {
        "upload_id": upload_id,
        "filename": Path(filename).name,
        "source": "user_uploaded_sevir_vil_image",
        "shape": [height, width],
        "dtype": "uint8",
        "grayscale_source": bool(grayscale_source),
        "kind": "single_frame",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if len(UPLOADED_VIL) >= 16:
        UPLOADED_VIL.pop(next(iter(UPLOADED_VIL)))
    UPLOADED_VIL[upload_id] = (frame.copy(), metadata)
    _persist_uploaded_array(upload_id, frame, metadata)
    return {
        "status": "ok",
        **metadata,
        "min_vil": int(frame.min()),
        "max_vil": int(frame.max()),
        "mean_vil": round(float(frame.mean()), 4),
        "preview_grid": _preview(frame),
        "note": (
            "已按原始灰度值读取。"
            if grayscale_source
            else "上传文件为彩色图，已转换为灰度强度；结果只能作为图像强度分析。"
        ),
    }


def _load_uploaded_vil(upload_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    if len(upload_id) != 32 or any(character not in "0123456789abcdef" for character in upload_id):
        raise ValueError("upload_id格式不正确。")
    uploaded = UPLOADED_VIL.get(upload_id)
    if uploaded is None:
        restored = _restore_uploaded_array(upload_id, expected_kind="single_frame")
        if restored is not None:
            uploaded = restored
            UPLOADED_VIL[upload_id] = restored
    if uploaded is None:
        raise ValueError("没有找到这次上传的SEVIR图片，请重新上传。")
    frame, metadata = uploaded
    if frame.shape != (384, 384) or frame.dtype != np.uint8:
        raise ValueError("上传样本的内部格式不正确。")
    return frame, metadata


def get_uploaded_display_frame(upload_id: str) -> np.ndarray:
    frame, _ = _load_uploaded_vil(upload_id)
    return frame.astype(np.float32)


def analyze_uploaded_vil(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    frame, metadata = _load_uploaded_vil(upload_id)
    mask = frame > threshold
    regions = [
        region
        for region in _hazard_regions(mask, metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_analysis",
        "analysis_scope": "single_uploaded_frame",
        "upload_id": upload_id,
        "filename": metadata["filename"],
        "data_source": metadata["source"],
        "model_name": "uploaded_vil_analysis",
        "spatial_shape": [384, 384],
        "threshold": threshold,
        "min_vil": int(frame.min()),
        "max_vil": int(frame.max()),
        "mean_vil": round(float(frame.mean()), 4),
        "exceeding_pixel_count": int(mask.sum()),
        "exceeding_area_ratio": round(float(mask.mean()), 6),
        "hazard_detected": bool(regions),
        "hazard_region_count": len(regions),
        "hazard_regions": regions,
        "full_resolution_viewer": True,
        "grayscale_source": metadata["grayscale_source"],
        "interpretation_limit": (
            "这是用户上传的单张SEVIR VIL图像，只能分析当前强回波与空间分布，"
            "不能据此声称完成未来预测或持续时间分析。阈值是0到255缩放像素值，"
            "不是dBZ、降雨量或官方灾害预警。"
        ),
    }


def register_uploaded_vil_sequence(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frames) != 13:
        raise ValueError("连续序列必须恰好包含13张384×384的SEVIR VIL灰度图。")
    decoded: list[np.ndarray] = []
    filenames: list[str] = []
    grayscale_flags: list[bool] = []
    for index, item in enumerate(frames):
        width = int(item.get("width", 0))
        height = int(item.get("height", 0))
        if (width, height) != (384, 384):
            raise ValueError(f"第{index + 1}张图不是384×384。")
        try:
            raw = base64.b64decode(str(item.get("pixels_base64", "")), validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"第{index + 1}张图的像素数据无法解析。") from error
        if len(raw) != 384 * 384:
            raise ValueError(f"第{index + 1}张图的像素数量不正确。")
        decoded.append(np.frombuffer(raw, dtype=np.uint8).reshape(384, 384).copy())
        filenames.append(Path(str(item.get("filename", f"frame_{index + 1:02d}.png"))).name)
        grayscale_flags.append(bool(item.get("grayscale_source", False)))
    sequence = np.stack(decoded, axis=0)
    upload_id = uuid.uuid4().hex
    metadata = {
        "upload_id": upload_id,
        "filenames": filenames,
        "source": "user_uploaded_sevir_vil_13_frame_sequence",
        "shape": [13, 384, 384],
        "dtype": "uint8",
        "all_grayscale_source": all(grayscale_flags),
        "history_minutes": list(range(-60, 5, 5)),
        "kind": "sequence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if len(UPLOADED_VIL_SEQUENCES) >= 8:
        UPLOADED_VIL_SEQUENCES.pop(next(iter(UPLOADED_VIL_SEQUENCES)))
    UPLOADED_VIL_SEQUENCES[upload_id] = (sequence, metadata)
    _persist_uploaded_array(upload_id, sequence, metadata)
    return {
        "status": "ok",
        **metadata,
        "frame_count": 13,
        "min_vil": int(sequence.min()),
        "max_vil": int(sequence.max()),
        "mean_vil": round(float(sequence.mean()), 4),
        "note": (
            "已按文件名排序读取13帧；可进行历史演变分析和Persistence基线，"
            "但任意序列现场CasCast仍需要EarthFormer确定性预测与latent压缩步骤。"
        ),
    }


def _load_uploaded_vil_sequence(upload_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    if len(upload_id) != 32 or any(character not in "0123456789abcdef" for character in upload_id):
        raise ValueError("sequence_upload_id格式不正确。")
    uploaded = UPLOADED_VIL_SEQUENCES.get(upload_id)
    if uploaded is None:
        restored = _restore_uploaded_array(upload_id, expected_kind="sequence")
        if restored is not None:
            uploaded = restored
            UPLOADED_VIL_SEQUENCES[upload_id] = restored
    if uploaded is None:
        raise ValueError("没有找到这次上传的13帧序列，请重新上传。")
    return uploaded


def get_uploaded_sequence_display_frame(
    upload_id: str, series: str, frame_index: int
) -> np.ndarray:
    sequence, _ = _load_uploaded_vil_sequence(upload_id)
    if series == "history":
        if not 0 <= frame_index < 13:
            raise IndexError("历史帧编号超出范围。")
        return sequence[frame_index].astype(np.float32)
    if series in {"forecast", "forecast_overview"}:
        if not 0 <= frame_index < 12 and series == "forecast":
            raise IndexError("预测帧编号超出范围。")
        return sequence[-1].astype(np.float32)
    raise ValueError(f"不支持的上传序列图像类型：{series}")


def analyze_uploaded_vil_sequence(
    upload_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    sequence, metadata = _load_uploaded_vil_sequence(upload_id)
    history_summaries = [
        {
            "frame_index": index + 1,
            "relative_minutes": int(metadata["history_minutes"][index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
            "exceeding_area_ratio": round(float((frame >= threshold).mean()), 6),
        }
        for index, frame in enumerate(sequence)
    ]
    persistence = np.repeat(sequence[-1][None, :, :], 12, axis=0).astype(np.float32)
    current_regions = [
        item for item in _hazard_regions(sequence[-1] >= threshold, metadata)
        if item["pixel_count"] >= minimum_region_pixels
    ]
    tracks = _storm_tracks(sequence.astype(np.float32), metadata, threshold, minimum_region_pixels)
    return {
        "status": "ok",
        "task": "uploaded_sevir_vil_sequence_analysis",
        "analysis_scope": "13_uploaded_history_frames_plus_persistence_baseline",
        "upload_id": upload_id,
        "data_source": metadata["source"],
        "model_name": "persistence_baseline",
        "input_frames": 13,
        "forecast_frames": 12,
        "lead_minutes": 60,
        "threshold": threshold,
        "history_available": True,
        "history_frames": 13,
        "history_frame_summaries": history_summaries,
        "frame_summaries": _prediction_frame_summaries(persistence.transpose(1, 2, 0), threshold),
        "ground_truth_available": False,
        "ground_truth_frame_summaries": [],
        "hazard_detected": bool(current_regions),
        "hazard_region_count": len(current_regions),
        "hazard_regions": current_regions,
        "historical_storm_tracks": tracks,
        "full_resolution_viewer": True,
        "spatial_shape": [384, 384],
        "interpretation_limit": (
            "13张图仅用于历史演变分析；网页给出的未来12帧是最后观测帧持续不变的Persistence基线，"
            "不是CasCast。要对任意上传序列现场运行CasCast，还需EarthFormer生成确定性未来、"
            "Autoencoder压缩latent后再送入CasCast概率模型。"
        ),
    }


def get_event_summary(event_id: str) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    return {
        "status": "ok",
        "data_source": _cascast_source() if model_name == "cascast" else _source(),
        "requested_event_id": event_id,
        "event_id": metadata["event_id"],
        "event_type": metadata.get("catalog_row", {}).get("event_type"),
        "channel": "VIL",
        "frame_count": int(vil.shape[-1]),
        "spatial_shape": [int(vil.shape[0]), int(vil.shape[1])],
        "min_vil": int(vil.min()),
        "max_vil": int(vil.max()),
        "mean_vil": round(float(vil.mean()), 4),
        "preview_grid": _preview(vil.max(axis=2)),
    }


def analyze_threshold(event_id: str, threshold: float) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    mask = vil > threshold
    per_frame_ratio = mask.mean(axis=(0, 1))
    peak_index = int(per_frame_ratio.argmax())
    return {
        "status": "ok",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "threshold": threshold,
        "pixel_frame_count": int(mask.size),
        "exceeding_pixel_frame_count": int(mask.sum()),
        "exceeding_ratio": round(float(mask.mean()), 6),
        "peak_frame_index": peak_index,
        "peak_frame_exceeding_ratio": round(float(per_frame_ratio[peak_index]), 6),
        "preview_grid": _preview(vil[:, :, peak_index]),
        "interpretation_limit": (
            "这里使用SEVIR文件内0到255的缩放VIL像素值。阈值40是数据级阈值，"
            "不能直接写成40 dBZ，也不能直接等同于地面降雨量。"
        ),
    }


def analyze_hazard(
    event_id: str,
    threshold: float,
    min_area_ratio: float = 0.001,
) -> dict[str, Any]:
    vil, metadata = _event(event_id)
    per_frame_ratio = (vil > threshold).mean(axis=(0, 1))
    active = per_frame_ratio >= min_area_ratio
    longest = 0
    current = 0
    end_index = -1
    for index, is_active in enumerate(active.tolist()):
        current = current + 1 if is_active else 0
        if current > longest:
            longest = current
            end_index = index
    start_index = end_index - longest + 1 if longest else -1
    minutes_per_frame = 5
    return {
        "status": "ok",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "analysis_scope": "observed_vil_frames",
        "threshold": threshold,
        "minimum_affected_area_ratio": min_area_ratio,
        "active_frame_count": int(active.sum()),
        "longest_active_start_frame": start_index,
        "longest_active_end_frame": end_index,
        "longest_active_duration_minutes": int(longest * minutes_per_frame),
        "peak_affected_area_ratio": round(float(per_frame_ratio.max()), 6),
        "preview_grid": _preview(vil[:, :, int(per_frame_ratio.argmax())]),
        "interpretation_limit": (
            "这是基于观测VIL阈值的强回波筛查，不是未来预报，也不是官方灾害定级。"
        ),
    }


def _binary_csi(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    valid = target != 255
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    true_positive = np.logical_and(predicted, observed).sum()
    union = np.logical_or(predicted, observed).sum()
    return float(true_positive / union) if union else 0.0


def _deterministic_metrics(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Evaluate raw CasCast ensemble mean using the official encoded VIL scale."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            f"预测和ground truth形状不一致：{prediction.shape} vs {target.shape}"
        )
    valid = target != 255
    if not valid.any():
        raise ValueError("Ground truth没有可用于评价的有效像素。")
    differences = prediction[valid] - target[valid]
    mse = float(np.mean(differences**2))
    mae = float(np.mean(np.abs(differences)))
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    hits = int(np.logical_and(predicted, observed).sum())
    misses = int(np.logical_and(~predicted, observed).sum())
    false_alarms = int(np.logical_and(predicted, ~observed & valid).sum())
    csi_denominator = hits + misses + false_alarms
    pod_denominator = hits + misses
    far_denominator = hits + false_alarms
    return {
        "ground_truth_available": True,
        "prediction_value_space": "raw CasCast VAE-decoded float on encoded VIL scale",
        "ground_truth_value_space": "SEVIR uint8 encoded VIL",
        "missing_value_code": 255,
        "valid_pixel_count": int(valid.sum()),
        "metric_threshold": threshold,
        "csi_at_threshold": round(hits / csi_denominator, 6) if csi_denominator else 0.0,
        "pod_at_threshold": round(hits / pod_denominator, 6) if pod_denominator else 0.0,
        "far_at_threshold": round(false_alarms / far_denominator, 6) if far_denominator else 0.0,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "mse_raw_vil": round(mse, 4),
        "rmse_raw_vil": round(float(np.sqrt(mse)), 4),
        "mae_raw_vil": round(mae, 4),
        "mse_normalized_0_1": round(mse / (255.0**2), 8),
        "postprocessing": "raw prediction; no clipping before metrics",
    }


def _longest_true_run(active: np.ndarray) -> tuple[int, int, int]:
    """Return start index, end index and length of the longest consecutive True run."""
    longest = 0
    current = 0
    end_index = -1
    for index, value in enumerate(np.asarray(active, dtype=bool).tolist()):
        current = current + 1 if value else 0
        if current > longest:
            longest = current
            end_index = index
    start_index = end_index - longest + 1 if longest else -1
    return start_index, end_index, longest


def _persistence_prediction(
    vil: np.ndarray, input_frames: int, forecast_frames: int
) -> np.ndarray:
    last_observation = vil[:, :, input_frames - 1].astype(np.float32)
    return np.repeat(last_observation[:, :, None], forecast_frames, axis=2)


def _hazard_regions(
    mask: np.ndarray, metadata: dict[str, Any], max_regions: int = 3
) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(mask):
        row_i, column_i = int(row), int(column)
        if visited[row_i, column_i]:
            continue
        queue = deque([(row_i, column_i)])
        visited[row_i, column_i] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        components.append(component)

    catalog = metadata.get("catalog_row", {})
    try:
        min_lat = float(catalog["llcrnrlat"])
        min_lon = float(catalog["llcrnrlon"])
        max_lat = float(catalog["urcrnrlat"])
        max_lon = float(catalog["urcrnrlon"])
    except (KeyError, TypeError, ValueError):
        min_lat = min_lon = max_lat = max_lon = None

    regions = []
    for rank, component in enumerate(
        sorted(components, key=len, reverse=True)[:max_regions], start=1
    ):
        rows = [point[0] for point in component]
        columns = [point[1] for point in component]
        center_row = sum(rows) / len(rows)
        center_column = sum(columns) / len(columns)
        region: dict[str, Any] = {
            "region_id": f"H{rank}",
            "pixel_count": len(component),
            "area_ratio": round(len(component) / mask.size, 6),
            "pixel_bbox": [min(columns), min(rows), max(columns), max(rows)],
            "pixel_centroid": [round(center_column, 1), round(center_row, 1)],
        }
        if min_lat is not None:
            longitude = min_lon + center_column / max(width - 1, 1) * (max_lon - min_lon)
            latitude = max_lat - center_row / max(height - 1, 1) * (max_lat - min_lat)
            region["approx_centroid_latlon"] = [round(latitude, 4), round(longitude, 4)]
            region["location_note"] = "由SEVIR事件边界线性映射得到的近似坐标"
        regions.append(region)
    return regions


def _prepare_prediction_data(
    event_id: str,
    model_name: str,
    input_frames: int,
    forecast_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, dict[str, Any], np.ndarray | None]:
    """Load the shared prediction material without formatting a public tool result."""
    if input_frames < 1 or forecast_frames < 1:
        raise ValueError("输入帧数和预测帧数必须大于0。")
    if model_name == "cascast":
        if forecast_frames != 12:
            raise ValueError("当前CasCast输出仅支持未来60分钟（12帧）；2/3小时滚动推理尚未执行。")
        arrays, metadata = _load_cascast_ensemble(event_id)
        prediction = arrays["ensemble_mean"].astype(np.float32).transpose(1, 2, 0)
        target = arrays["ground_truth"].astype(np.float32).transpose(1, 2, 0)
        return prediction, arrays, metadata, target
    if model_name != "persistence_baseline":
        raise ValueError(f"不支持的预测模型：{model_name}")
    try:
        arrays, metadata = _load_cascast_ensemble(event_id)
    except ValueError:
        arrays = None
    if arrays is not None:
        history = arrays["history"].astype(np.float32)
        if input_frames > history.shape[0]:
            raise ValueError("当前CasCast事件没有足够的历史输入帧。")
        prediction = np.repeat(history[input_frames - 1][:, :, None], forecast_frames, axis=2)
        target = arrays["ground_truth"].astype(np.float32).transpose(1, 2, 0)
        return prediction, arrays, metadata, target
    vil, metadata = _event(event_id)
    if input_frames + forecast_frames > vil.shape[-1]:
        raise ValueError("当前事件没有足够帧数用于指定的输入与验证区间。")
    prediction = _persistence_prediction(vil, input_frames, forecast_frames)
    target = vil[:, :, input_frames : input_frames + forecast_frames].astype(np.float32)
    return prediction, None, metadata, target


def _prediction_frame_summaries(prediction: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": index + 1,
            "lead_minutes": (index + 1) * 5,
            "min_vil": round(float(frame.min()), 3),
            "max_vil": round(float(frame.max()), 3),
            "mean_vil": round(float(frame.mean()), 3),
            "exceeding_area_ratio": round(float((frame >= threshold).mean()), 6),
            "area_threshold": threshold,
        }
        for index, frame in enumerate(prediction.transpose(2, 0, 1))
    ]


def _history_frame_summaries(arrays: dict[str, np.ndarray] | None) -> list[dict[str, Any]]:
    if arrays is None:
        return []
    history_minutes = _ensemble_manifest().get("history_minutes", list(range(-60, 5, 5)))
    return [
        {
            "frame_index": index + 1,
            "relative_minutes": int(history_minutes[index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
        }
        for index, frame in enumerate(arrays["history"])
    ]


def run_nowcast(
    event_id: str,
    model_name: str,
    threshold: float,
    input_frames: int = 13,
    forecast_frames: int = 12,
) -> dict[str, Any]:
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, input_frames, forecast_frames
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    if arrays is not None:
        result = {
            "status": "ok",
            "task": "nowcast",
            "data_source": _cascast_source(str(metadata["event_id"])),
            "event_id": metadata["event_id"],
            "model_name": model_name,
            "input_frames": input_frames,
            "forecast_frames": forecast_frames,
            "lead_minutes": forecast_frames * 5,
            "inference_mode": (
                "precomputed_multi_event_ensemble"
                if model_name == "cascast"
                else "persistence_from_matching_cascast_history"
            ),
            "ensemble_count": metadata.get("ensemble_count", 10) if model_name == "cascast" else None,
            "ground_truth_available": target is not None,
            "history_available": True,
            "history_frames": int(arrays["history"].shape[0]),
            "history_frame_summaries": _history_frame_summaries(arrays),
            "full_resolution_viewer": True,
            "spatial_shape": [384, 384],
            "min_vil": round(float(analysis_prediction.min()), 4),
            "max_vil": round(float(analysis_prediction.max()), 4),
            "mean_vil": round(float(analysis_prediction.mean()), 4),
            "frame_summaries": frame_summaries,
            "ground_truth_frame_summaries": _prediction_frame_summaries(target, threshold) if target is not None else [],
            "model_metrics": _deterministic_metrics(prediction, target, threshold) if target is not None else {"ground_truth_available": False},
            "interpretation_limit": "指标仅评价当前所选SEVIR测试样本；不能代表完整测试集总体性能。",
        }
        if model_name == "cascast":
            result.update(
                {
                    "available_event_count": int(_ensemble_manifest()["samples"]),
                    "code_available": (CASCAST_ROOT / "networks" / "casformer.py").exists(),
                    "checkpoint_available": (CASCAST_ROOT / "experiments" / "cascast_diffusion" / "world_size1-ckpt" / "checkpoint_best.pth").exists(),
                    "autoencoder_checkpoint_available": (CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth").exists(),
                }
            )
        return result

    assert target is not None
    mse_raw = float(np.mean((prediction - target) ** 2))
    return {
        "status": "ok",
        "task": "nowcast",
        "data_source": _source(),
        "event_id": metadata["event_id"],
        "model_name": model_name,
        "input_frames": input_frames,
        "forecast_frames": forecast_frames,
        "ground_truth_available": True,
        "evaluation_target": "同一事件中紧随输入序列的真实VIL帧",
        "metric_threshold": threshold,
        "csi_at_threshold": round(_binary_csi(prediction, target, threshold), 6),
        "mse_raw_vil": round(mse_raw, 4),
        "mse_normalized_0_1": round(mse_raw / (255.0**2), 8),
        "frame_summaries": frame_summaries,
        "interpretation_limit": "这是可复现的persistence基线，不是CasCast结果。",
    }


def forecast_hazards(
    event_id: str,
    model_name: str,
    threshold: float,
    lead_minutes: int = 60,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    if lead_minutes not in {60, 120, 180}:
        raise ValueError("当前Demo支持未来1、2或3小时预测时效。")
    forecast_frames = lead_minutes // 5
    prediction, arrays, metadata, target = _prepare_prediction_data(
        event_id, model_name, 13, forecast_frames
    )
    analysis_prediction = _vil_display_values(prediction)
    frame_summaries = _prediction_frame_summaries(analysis_prediction, threshold)
    hazard_mask = analysis_prediction.max(axis=2) >= threshold
    regions = [
        region
        for region in _hazard_regions(hazard_mask, metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    for region in regions:
        min_x, min_y, max_x, max_y = region["pixel_bbox"]
        region_active = (
            analysis_prediction[min_y : max_y + 1, min_x : max_x + 1, :] >= threshold
        ).any(axis=(0, 1))
        start_index, end_index, longest = _longest_true_run(region_active)
        if longest:
            region["start_minute"] = int((start_index + 1) * 5)
            region["end_minute"] = int((end_index + 1) * 5)
            region["duration_minutes"] = int(longest * 5)
            region["total_active_minutes"] = int(region_active.sum() * 5)
            region["duration_is_lower_bound"] = bool(region_active.all())

    exceeding_pixel_counts = (analysis_prediction >= threshold).sum(axis=(0, 1))
    any_pixel_active = exceeding_pixel_counts > 0
    hazard_area_active = exceeding_pixel_counts >= minimum_region_pixels
    _, _, any_pixel_longest = _longest_true_run(any_pixel_active)
    _, _, hazard_longest = _longest_true_run(hazard_area_active)
    duration = int(hazard_longest * 5)
    return {
        "status": "ok",
        "data_source": _cascast_source(str(metadata["event_id"])) if arrays is not None else _source(),
        "event_id": metadata["event_id"],
        "task": "future_hazard_analysis",
        "model_name": model_name,
        "lead_minutes": lead_minutes,
        "threshold": threshold,
        "hazard_detected": bool(regions),
        "hazard_region_count": len(regions),
        "hazard_regions": regions,
        "threshold_exceeded": bool(any_pixel_active.any()),
        "maximum_exceeding_pixel_count": int(exceeding_pixel_counts.max()),
        "any_pixel_exceedance_total_minutes": int(any_pixel_active.sum() * 5),
        "any_pixel_exceedance_longest_consecutive_minutes": int(any_pixel_longest * 5),
        "hazard_area_minimum_pixels": minimum_region_pixels,
        "hazard_area_total_minutes": int(hazard_area_active.sum() * 5),
        "strong_echo_duration_minutes_within_horizon": duration,
        "duration_is_lower_bound": bool(hazard_area_active.all()),
        "frame_summaries": frame_summaries,
        "ground_truth_frame_summaries": _prediction_frame_summaries(target, threshold) if target is not None else [],
        "history_available": arrays is not None,
        "history_frames": int(arrays["history"].shape[0]) if arrays is not None else 0,
        "history_frame_summaries": _history_frame_summaries(arrays),
        "full_resolution_viewer": arrays is not None,
        "spatial_shape": [384, 384],
        "ensemble_count": metadata.get("ensemble_count") if model_name == "cascast" else None,
        "model_metrics": (
            {
                **_deterministic_metrics(prediction, target, threshold),
            }
            if target is not None
            else {"ground_truth_available": False}
        ),
        "interpretation_limit": (
            f"危险区域来自{model_name}预测的VIL强回波代理。"
            f"阈值{threshold:g}是0至255范围的缩放像素值，不是dBZ或降雨量；"
            "这不是官方灾害预警。"
        ),
    }


def analyze_ensemble_uncertainty(
    event_id: str,
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    arrays, metadata = _load_cascast_ensemble(event_id)
    members = arrays["ensemble_members"].astype(np.float32)
    mean_prediction = arrays["ensemble_mean"]
    ground_truth = arrays["ground_truth"].astype(np.float32)
    pixel_std = arrays["pixel_std"]
    exceedance_probability = (members >= threshold).mean(axis=0)

    member_summaries: list[dict[str, Any]] = []
    object_centroids: list[list[float]] = []
    largest_object_area_ratios: list[float] = []
    for member_index, member in enumerate(members):
        per_frame_area = (member >= threshold).mean(axis=(1, 2))
        active_frames = (member >= threshold).sum(axis=(1, 2)) >= minimum_region_pixels
        regions = [
            region
            for region in _hazard_regions((member >= threshold).any(axis=0), metadata, max_regions=1)
            if region["pixel_count"] >= minimum_region_pixels
        ]
        largest = regions[0] if regions else None
        if largest:
            object_centroids.append(largest["pixel_centroid"])
            largest_object_area_ratios.append(float(largest["area_ratio"]))
        member_summaries.append(
            {
                "member": member_index + 1,
                "seed": int(arrays["member_seed"][member_index]),
                "max_vil": round(float(member.max()), 3),
                "mean_vil": round(float(member.mean()), 3),
                "peak_exceeding_area_ratio": round(float(per_frame_area.max()), 6),
                "active_duration_minutes": int(active_frames.sum() * 5),
                "largest_object_area_ratio": round(float(largest["area_ratio"]), 6) if largest else 0.0,
                "largest_object_centroid": largest["pixel_centroid"] if largest else None,
            }
        )

    if object_centroids:
        centroid_array = np.asarray(object_centroids, dtype=np.float32)
        centroid_spread = float(np.sqrt(np.var(centroid_array, axis=0).sum()))
    else:
        centroid_spread = 0.0

    uncertainty_frames = []
    for frame_index in range(12):
        frame_mean = mean_prediction[frame_index]
        frame_std = pixel_std[frame_index]
        frame_probability = exceedance_probability[frame_index]
        uncertainty_frames.append(
            {
                "frame_index": frame_index + 1,
                "lead_minutes": (frame_index + 1) * 5,
                "mean_vil": round(float(frame_mean.mean()), 3),
                "max_mean_vil": round(float(frame_mean.max()), 3),
                "max_vil": round(float(frame_mean.max()), 3),
                "mean_pixel_std": round(float(frame_std.mean()), 3),
                "p95_pixel_std": round(float(np.quantile(frame_std, 0.95)), 3),
                "expected_exceeding_area_ratio": round(float(frame_probability.mean()), 6),
                "exceeding_area_ratio": round(float(frame_probability.mean()), 6),
                "area_threshold": threshold,
                "max_exceedance_probability": round(float(frame_probability.max()), 3),
            }
        )

    member_areas = np.asarray(
        [item["peak_exceeding_area_ratio"] for item in member_summaries], dtype=np.float32
    )
    member_durations = np.asarray(
        [item["active_duration_minutes"] for item in member_summaries], dtype=np.float32
    )
    mean_field = mean_prediction.transpose(1, 2, 0)
    mean_regions = [
        region
        for region in _hazard_regions((mean_field >= threshold).any(axis=2), metadata)
        if region["pixel_count"] >= minimum_region_pixels
    ]
    for region in mean_regions:
        min_x, min_y, max_x, max_y = region["pixel_bbox"]
        active = (mean_prediction[:, min_y : max_y + 1, min_x : max_x + 1] >= threshold).any(axis=(1, 2))
        start_index, end_index, longest = _longest_true_run(active)
        if longest:
            region["start_minute"] = int((start_index + 1) * 5)
            region["end_minute"] = int((end_index + 1) * 5)
            region["duration_minutes"] = int(longest * 5)
            region["total_active_minutes"] = int(active.sum() * 5)
            region["duration_is_lower_bound"] = bool(active.all())
    history_minutes = _ensemble_manifest().get("history_minutes", list(range(-60, 5, 5)))
    history_summaries = [
        {
            "frame_index": index + 1,
            "relative_minutes": int(history_minutes[index]),
            "min_vil": int(frame.min()),
            "max_vil": int(frame.max()),
            "mean_vil": round(float(frame.mean()), 3),
        }
        for index, frame in enumerate(arrays["history"])
    ]
    return {
        "status": "ok",
        "task": "ensemble_uncertainty_analysis",
        "data_source": _cascast_source(str(metadata["event_id"])),
        "event_id": metadata["event_id"],
        "model_name": "cascast",
        "lead_minutes": 60,
        "forecast_frames": 12,
        "ensemble_count": int(members.shape[0]),
        "threshold": threshold,
        "ground_truth_available": True,
        "history_available": True,
        "history_frames": int(arrays["history"].shape[0]),
        "history_frame_summaries": history_summaries,
        "full_resolution_viewer": True,
        "spatial_shape": [384, 384],
        "hazard_regions": mean_regions,
        "ground_truth_frame_summaries": _prediction_frame_summaries(
            ground_truth.transpose(1, 2, 0), threshold
        ),
        "model_metrics": _deterministic_metrics(
            mean_prediction.transpose(1, 2, 0),
            ground_truth.transpose(1, 2, 0),
            threshold,
        ),
        "pixel_uncertainty": {
            "mean_std": round(float(pixel_std.mean()), 3),
            "p95_std": round(float(np.quantile(pixel_std, 0.95)), 3),
            "max_std": round(float(pixel_std.max()), 3),
            "mean_80_percent_interval_width": round(float((arrays["q90"] - arrays["q10"]).mean()), 3),
        },
        "storm_object_uncertainty": {
            "members_with_object": len(object_centroids),
            "centroid_spread_pixels": round(centroid_spread, 3),
            "largest_object_area_ratio_min": round(min(largest_object_area_ratios), 6) if largest_object_area_ratios else 0.0,
            "largest_object_area_ratio_mean": round(float(np.mean(largest_object_area_ratios)), 6) if largest_object_area_ratios else 0.0,
            "largest_object_area_ratio_max": round(max(largest_object_area_ratios), 6) if largest_object_area_ratios else 0.0,
        },
        "member_peak_area_ratio": {
            "min": round(float(member_areas.min()), 6),
            "mean": round(float(member_areas.mean()), 6),
            "max": round(float(member_areas.max()), 6),
        },
        "member_active_duration_minutes": {
            "min": int(member_durations.min()),
            "mean": round(float(member_durations.mean()), 1),
            "max": int(member_durations.max()),
        },
        "member_summaries": member_summaries,
        "frame_summaries": uncertainty_frames,
        "uncertainty_frame_summaries": uncertainty_frames,
        "interpretation_limit": (
            "不确定性来自10个CasCast随机ensemble成员之间的离散程度；"
            "它描述模型预测分歧，不等于真实误差或灾害概率。"
            "准确率指标只对应当前所选测试样本。"
        ),
    }


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "valid": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "false_alarms": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
    }


def _accumulate_metrics(
    accumulator: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> None:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = target != 255
    difference = prediction[valid] - target[valid]
    predicted = (prediction >= threshold) & valid
    observed = (target >= threshold) & valid
    accumulator["valid"] += float(valid.sum())
    accumulator["hits"] += float(np.logical_and(predicted, observed).sum())
    accumulator["misses"] += float(np.logical_and(~predicted, observed).sum())
    accumulator["false_alarms"] += float(
        np.logical_and(predicted, ~observed & valid).sum()
    )
    accumulator["squared_error"] += float(np.square(difference).sum(dtype=np.float64))
    accumulator["absolute_error"] += float(np.abs(difference).sum(dtype=np.float64))


def _finalize_metrics(accumulator: dict[str, float], threshold: float) -> dict[str, Any]:
    valid = int(accumulator["valid"])
    hits = int(accumulator["hits"])
    misses = int(accumulator["misses"])
    false_alarms = int(accumulator["false_alarms"])
    mse = accumulator["squared_error"] / valid if valid else 0.0
    mae = accumulator["absolute_error"] / valid if valid else 0.0
    return {
        "metric_threshold": float(threshold),
        "valid_pixel_count": valid,
        "csi": round(hits / (hits + misses + false_alarms), 6)
        if hits + misses + false_alarms
        else 0.0,
        "pod": round(hits / (hits + misses), 6) if hits + misses else 0.0,
        "far": round(false_alarms / (hits + false_alarms), 6)
        if hits + false_alarms
        else 0.0,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "mse_raw_vil": round(mse, 4),
        "rmse_raw_vil": round(float(np.sqrt(mse)), 4),
        "mae_raw_vil": round(mae, 4),
        "mse_normalized_0_1": round(mse / (255.0**2), 8),
    }


def _bbox_iou(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right < left or bottom < top:
        return 0.0
    intersection = (right - left + 1) * (bottom - top + 1)
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    second_area = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    return float(intersection / (first_area + second_area - intersection))


def _storm_tracks(
    fields: np.ndarray,
    metadata: dict[str, Any],
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    """Track thresholded connected regions across consecutive five-minute frames."""
    fields = np.asarray(fields, dtype=np.float32)
    tracks: list[dict[str, Any]] = []
    active_track_ids: list[int] = []
    frame_object_counts: list[int] = []
    for frame_index, frame in enumerate(fields):
        objects = [
            region
            for region in _hazard_regions(frame >= threshold, metadata, max_regions=20)
            if region["pixel_count"] >= minimum_region_pixels
        ]
        frame_object_counts.append(len(objects))
        candidates: list[tuple[float, float, int, int]] = []
        for track_id in active_track_ids:
            previous = tracks[track_id]["objects"][-1]
            for object_index, region in enumerate(objects):
                iou = _bbox_iou(previous["pixel_bbox"], region["pixel_bbox"])
                distance = float(
                    np.linalg.norm(
                        np.asarray(previous["pixel_centroid"], dtype=np.float32)
                        - np.asarray(region["pixel_centroid"], dtype=np.float32)
                    )
                )
                if iou > 0 or distance <= 32.0:
                    candidates.append((iou, -distance, track_id, object_index))
        matched_tracks: set[int] = set()
        matched_objects: set[int] = set()
        for _, _, track_id, object_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or object_index in matched_objects:
                continue
            tracks[track_id]["objects"].append(objects[object_index])
            tracks[track_id]["frame_indices"].append(frame_index)
            matched_tracks.add(track_id)
            matched_objects.add(object_index)
        new_active = list(matched_tracks)
        for object_index, region in enumerate(objects):
            if object_index in matched_objects:
                continue
            tracks.append({"frame_indices": [frame_index], "objects": [region]})
            new_active.append(len(tracks) - 1)
        active_track_ids = new_active

    public_tracks: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks, start=1):
        frames = track["frame_indices"]
        objects = track["objects"]
        centroids = np.asarray([item["pixel_centroid"] for item in objects], dtype=np.float32)
        path_length = float(
            np.linalg.norm(np.diff(centroids, axis=0), axis=1).sum()
        ) if len(centroids) > 1 else 0.0
        public_tracks.append(
            {
                "track_id": f"T{track_index}",
                "start_minute": int((frames[0] + 1) * 5),
                "end_minute": int((frames[-1] + 1) * 5),
                "duration_minutes": int(len(frames) * 5),
                "frame_count": len(frames),
                "maximum_pixel_count": max(item["pixel_count"] for item in objects),
                "maximum_area_ratio": round(max(item["area_ratio"] for item in objects), 6),
                "path_length_pixels": round(path_length, 3),
                "start_centroid": objects[0]["pixel_centroid"],
                "end_centroid": objects[-1]["pixel_centroid"],
            }
        )
    public_tracks.sort(key=lambda item: (item["duration_minutes"], item["maximum_pixel_count"]), reverse=True)
    return {
        "track_count": len(public_tracks),
        "frame_object_counts": frame_object_counts,
        "longest_track_duration_minutes": max(
            (item["duration_minutes"] for item in public_tracks), default=0
        ),
        "tracks": public_tracks[:10],
        "association_method": "consecutive-frame connected components matched by bbox overlap or centroid distance <=32 pixels",
    }


def _storm_object_verification(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any],
    threshold: float,
    minimum_region_pixels: int = 16,
) -> dict[str, Any]:
    hits = misses = false_alarms = 0
    centroid_errors: list[float] = []
    per_frame: list[dict[str, Any]] = []
    for frame_index, (predicted_frame, target_frame) in enumerate(zip(prediction, target)):
        predicted_objects = [
            item for item in _hazard_regions(predicted_frame >= threshold, metadata, max_regions=20)
            if item["pixel_count"] >= minimum_region_pixels
        ]
        target_objects = [
            item for item in _hazard_regions(target_frame >= threshold, metadata, max_regions=20)
            if item["pixel_count"] >= minimum_region_pixels
        ]
        candidates: list[tuple[float, int, int]] = []
        for pred_index, predicted_object in enumerate(predicted_objects):
            for target_index, target_object in enumerate(target_objects):
                iou = _bbox_iou(predicted_object["pixel_bbox"], target_object["pixel_bbox"])
                if iou >= 0.1:
                    candidates.append((iou, pred_index, target_index))
        matched_predicted: set[int] = set()
        matched_target: set[int] = set()
        for _, pred_index, target_index in sorted(candidates, reverse=True):
            if pred_index in matched_predicted or target_index in matched_target:
                continue
            matched_predicted.add(pred_index)
            matched_target.add(target_index)
            centroid_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(predicted_objects[pred_index]["pixel_centroid"], dtype=np.float32)
                        - np.asarray(target_objects[target_index]["pixel_centroid"], dtype=np.float32)
                    )
                )
            )
        frame_hits = len(matched_predicted)
        frame_misses = len(target_objects) - frame_hits
        frame_false_alarms = len(predicted_objects) - frame_hits
        hits += frame_hits
        misses += frame_misses
        false_alarms += frame_false_alarms
        per_frame.append(
            {
                "lead_minutes": (frame_index + 1) * 5,
                "predicted_objects": len(predicted_objects),
                "observed_objects": len(target_objects),
                "hits": frame_hits,
                "misses": frame_misses,
                "false_alarms": frame_false_alarms,
            }
        )
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "object_csi": round(hits / (hits + misses + false_alarms), 6)
        if hits + misses + false_alarms else 0.0,
        "object_pod": round(hits / (hits + misses), 6) if hits + misses else 0.0,
        "object_far": round(false_alarms / (hits + false_alarms), 6)
        if hits + false_alarms else 0.0,
        "mean_matched_centroid_error_pixels": round(float(np.mean(centroid_errors)), 3)
        if centroid_errors else None,
        "per_frame": per_frame,
        "matching_method": "connected regions matched one-to-one using bounding-box IoU >= 0.1",
    }


def build_cascast_evaluation_report(primary_threshold: float = 133.0) -> dict[str, Any]:
    """Evaluate all locally available CasCast events against their aligned ground truth."""
    manifest = _ensemble_manifest()
    thresholds = list(SEVIR_EVALUATION_THRESHOLDS)
    if float(primary_threshold) not in thresholds:
        thresholds.append(float(primary_threshold))
        thresholds.sort()
    model_accumulators = {
        model: {threshold: _empty_metric_accumulator() for threshold in thresholds}
        for model in ("cascast_ensemble_mean", "persistence")
    }
    lead_accumulators = {
        model: {
            threshold: [_empty_metric_accumulator() for _ in range(12)]
            for threshold in thresholds
        }
        for model in ("cascast_ensemble_mean", "persistence")
    }
    member_accumulators = [_empty_metric_accumulator() for _ in range(int(manifest["members_per_sample"]))]
    reliability = {
        threshold: [{"count": 0, "observed": 0} for _ in range(11)]
        for threshold in thresholds
    }
    rank_histogram = np.zeros(int(manifest["members_per_sample"]) + 1, dtype=np.int64)
    coverage_count = interval_valid = 0
    interval_width_sum = 0.0
    spread_sum = squared_error_sum = 0.0
    correlation = {key: 0.0 for key in ("n", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")}
    event_results: list[dict[str, Any]] = []
    aggregate_object = {"hits": 0, "misses": 0, "false_alarms": 0, "centroid_error_sum": 0.0, "centroid_error_count": 0}

    for event in manifest["events"]:
        arrays, metadata = _load_cascast_ensemble(event["event_id"])
        members = arrays["ensemble_members"].astype(np.float32)
        mean_prediction = arrays["ensemble_mean"].astype(np.float32)
        target = arrays["ground_truth"].astype(np.float32)
        persistence = np.repeat(arrays["history"][-1][None, :, :], target.shape[0], axis=0).astype(np.float32)
        valid = target != 255

        for threshold in thresholds:
            for model, prediction in (
                ("cascast_ensemble_mean", mean_prediction),
                ("persistence", persistence),
            ):
                _accumulate_metrics(model_accumulators[model][threshold], prediction, target, threshold)
                for frame_index in range(target.shape[0]):
                    _accumulate_metrics(
                        lead_accumulators[model][threshold][frame_index],
                        prediction[frame_index],
                        target[frame_index],
                        threshold,
                    )
            probability = (members >= threshold).mean(axis=0)
            observed = target >= threshold
            probability_index = np.clip(np.rint(probability * 10).astype(np.int8), 0, 10)
            for bin_index in range(11):
                selected = (probability_index == bin_index) & valid
                reliability[threshold][bin_index]["count"] += int(selected.sum())
                reliability[threshold][bin_index]["observed"] += int(observed[selected].sum())

        for member_index, member in enumerate(members):
            _accumulate_metrics(member_accumulators[member_index], member, target, primary_threshold)

        q10 = arrays["q10"].astype(np.float32)
        q90 = arrays["q90"].astype(np.float32)
        interval_valid += int(valid.sum())
        coverage_count += int(((target >= q10) & (target <= q90) & valid).sum())
        interval_width_sum += float((q90 - q10)[valid].sum(dtype=np.float64))
        std = arrays["pixel_std"].astype(np.float32)[valid].astype(np.float64)
        absolute_error = np.abs(mean_prediction[valid] - target[valid]).astype(np.float64)
        spread_sum += float(std.sum())
        squared_error_sum += float(np.square(mean_prediction[valid] - target[valid]).sum(dtype=np.float64))
        correlation["n"] += float(std.size)
        correlation["sum_x"] += float(std.sum())
        correlation["sum_y"] += float(absolute_error.sum())
        correlation["sum_xx"] += float(np.square(std).sum())
        correlation["sum_yy"] += float(np.square(absolute_error).sum())
        correlation["sum_xy"] += float((std * absolute_error).sum())
        ranks = np.sum(members < target[None, :, :, :], axis=0)
        rank_histogram += np.bincount(ranks[valid].ravel(), minlength=len(rank_histogram))

        predicted_tracks = _storm_tracks(mean_prediction, metadata, primary_threshold)
        observed_tracks = _storm_tracks(target, metadata, primary_threshold)
        object_verification = _storm_object_verification(
            mean_prediction, target, metadata, primary_threshold
        )
        for key in ("hits", "misses", "false_alarms"):
            aggregate_object[key] += int(object_verification[key])
        if object_verification["mean_matched_centroid_error_pixels"] is not None:
            matched = int(object_verification["hits"])
            aggregate_object["centroid_error_sum"] += float(
                object_verification["mean_matched_centroid_error_pixels"] * matched
            )
            aggregate_object["centroid_error_count"] += matched
        event_results.append(
            {
                "event_id": metadata["event_id"],
                "sample_index": metadata["sample_index"],
                "event_type": metadata.get("event_type") or "Unlabeled",
                "cascast": _deterministic_metrics(
                    mean_prediction.transpose(1, 2, 0),
                    target.transpose(1, 2, 0),
                    primary_threshold,
                ),
                "persistence": _deterministic_metrics(
                    persistence.transpose(1, 2, 0),
                    target.transpose(1, 2, 0),
                    primary_threshold,
                ),
                "predicted_storm_tracks": predicted_tracks,
                "observed_storm_tracks": observed_tracks,
                "storm_object_verification": object_verification,
            }
        )

    n = correlation["n"]
    numerator = n * correlation["sum_xy"] - correlation["sum_x"] * correlation["sum_y"]
    denominator = np.sqrt(
        max(n * correlation["sum_xx"] - correlation["sum_x"] ** 2, 0.0)
        * max(n * correlation["sum_yy"] - correlation["sum_y"] ** 2, 0.0)
    )
    member_metrics = [
        _finalize_metrics(accumulator, primary_threshold) for accumulator in member_accumulators
    ]
    member_metric_summary = {
        key: {
            "min": round(min(float(item[key]) for item in member_metrics), 6),
            "mean": round(float(np.mean([float(item[key]) for item in member_metrics])), 6),
            "max": round(max(float(item[key]) for item in member_metrics), 6),
        }
        for key in ("csi", "pod", "far", "mse_raw_vil", "mae_raw_vil")
    }
    object_denominator = aggregate_object["hits"] + aggregate_object["misses"] + aggregate_object["false_alarms"]
    object_pod_denominator = aggregate_object["hits"] + aggregate_object["misses"]
    object_far_denominator = aggregate_object["hits"] + aggregate_object["false_alarms"]
    report = {
        "status": "ok",
        "task": "cascast_dataset_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "10 aligned SEVIR VIL test events with CasCast ensembles and ground truth",
        "event_count": len(event_results),
        "ensemble_count": int(manifest["members_per_sample"]),
        "forecast_frames": 12,
        "lead_minutes": 60,
        "primary_threshold": float(primary_threshold),
        "thresholds": thresholds,
        "aggregate_metrics": {
            model: {
                str(int(threshold) if threshold.is_integer() else threshold): _finalize_metrics(accumulator, threshold)
                for threshold, accumulator in by_threshold.items()
            }
            for model, by_threshold in model_accumulators.items()
        },
        "lead_time_metrics": {
            model: {
                str(int(threshold) if threshold.is_integer() else threshold): [
                    {"lead_minutes": (index + 1) * 5, **_finalize_metrics(accumulator, threshold)}
                    for index, accumulator in enumerate(accumulators)
                ]
                for threshold, accumulators in by_threshold.items()
            }
            for model, by_threshold in lead_accumulators.items()
        },
        "ensemble_member_performance": {
            "threshold": float(primary_threshold),
            "members": [
                {"member": index + 1, **metrics} for index, metrics in enumerate(member_metrics)
            ],
            "summary": member_metric_summary,
        },
        "ensemble_validation": {
            "mean_pixel_std": round(spread_sum / interval_valid, 4) if interval_valid else 0.0,
            "ensemble_mean_rmse": round(np.sqrt(squared_error_sum / interval_valid), 4) if interval_valid else 0.0,
            "spread_skill_ratio": round(
                (spread_sum / interval_valid) / np.sqrt(squared_error_sum / interval_valid), 6
            ) if interval_valid and squared_error_sum else 0.0,
            "pixel_std_absolute_error_pearson": round(float(numerator / denominator), 6)
            if denominator else 0.0,
            "central_80_interval_coverage": round(coverage_count / interval_valid, 6)
            if interval_valid else 0.0,
            "mean_central_80_interval_width": round(interval_width_sum / interval_valid, 4)
            if interval_valid else 0.0,
            "rank_histogram": rank_histogram.astype(int).tolist(),
            "rank_bins": list(range(len(rank_histogram))),
            "reliability": {
                str(int(threshold) if threshold.is_integer() else threshold): [
                    {
                        "forecast_probability": bin_index / 10,
                        "pixel_count": item["count"],
                        "observed_frequency": round(item["observed"] / item["count"], 6)
                        if item["count"] else None,
                    }
                    for bin_index, item in enumerate(bins)
                ]
                for threshold, bins in reliability.items()
            },
            "interpretation": (
                "spread-error相关、区间覆盖率、spread-skill ratio、可靠性曲线和rank histogram"
                "均由10路ensemble与真实未来帧共同计算；它们比只展示标准差更能检验不确定性是否可信。"
            ),
        },
        "storm_object_evaluation": {
            "threshold": float(primary_threshold),
            "minimum_region_pixels": 16,
            "hits": aggregate_object["hits"],
            "misses": aggregate_object["misses"],
            "false_alarms": aggregate_object["false_alarms"],
            "object_csi": round(aggregate_object["hits"] / object_denominator, 6)
            if object_denominator else 0.0,
            "object_pod": round(aggregate_object["hits"] / object_pod_denominator, 6)
            if object_pod_denominator else 0.0,
            "object_far": round(aggregate_object["false_alarms"] / object_far_denominator, 6)
            if object_far_denominator else 0.0,
            "mean_matched_centroid_error_pixels": round(
                aggregate_object["centroid_error_sum"] / aggregate_object["centroid_error_count"], 3
            ) if aggregate_object["centroid_error_count"] else None,
            "scope": "connected-region tracks and framewise bbox-IoU object verification",
        },
        "events": event_results,
        "interpretation_limit": (
            "这是当前本地10个测试事件的聚合评价，不是SEVIR完整测试集论文复现结果；"
            "storm-object匹配使用连通域外接框IoU和中心距离，尚未处理复杂分裂与合并。"
        ),
    }
    return report


def _evaluation_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    threshold_key = str(int(float(report["primary_threshold"])))
    validation = dict(report["ensemble_validation"])
    reliability = validation.pop("reliability", {})
    validation["reliability_at_primary_threshold"] = reliability.get(threshold_key, [])
    events = []
    for event in report["events"]:
        verification = dict(event["storm_object_verification"])
        verification.pop("per_frame", None)
        events.append(
            {
                "event_id": event["event_id"],
                "sample_index": event["sample_index"],
                "event_type": event["event_type"],
                "cascast": event["cascast"],
                "persistence": event["persistence"],
                "predicted_track_count": event["predicted_storm_tracks"]["track_count"],
                "observed_track_count": event["observed_storm_tracks"]["track_count"],
                "longest_predicted_track_minutes": event["predicted_storm_tracks"]["longest_track_duration_minutes"],
                "longest_observed_track_minutes": event["observed_storm_tracks"]["longest_track_duration_minutes"],
                "storm_object_verification": verification,
            }
        )
    return {
        key: report[key]
        for key in (
            "status", "task", "generated_at_utc", "data_source", "event_count",
            "ensemble_count", "forecast_frames", "lead_minutes", "primary_threshold",
            "thresholds", "aggregate_metrics", "ensemble_member_performance",
            "storm_object_evaluation", "interpretation_limit",
        )
    } | {
        "lead_time_metrics": {
            model: values[threshold_key] for model, values in report["lead_time_metrics"].items()
        },
        "ensemble_validation": validation,
        "events": events,
        "full_report_file": str(CASCAST_EVALUATION_REPORT.relative_to(ROOT)),
    }


def evaluate_cascast_dataset(primary_threshold: float = 133.0) -> dict[str, Any]:
    """Load a reproducible cached report and return an agent-friendly summary."""
    if CASCAST_EVALUATION_REPORT.exists():
        report = json.loads(CASCAST_EVALUATION_REPORT.read_text(encoding="utf-8"))
        if float(report.get("primary_threshold", -1)) == float(primary_threshold):
            return _evaluation_report_summary(report)
    report = build_cascast_evaluation_report(primary_threshold)
    CASCAST_EVALUATION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _evaluation_report_summary(report)


def run_satellite_to_radar(event_id: str) -> dict[str, Any]:
    _, metadata = _event(event_id)
    return {
        "status": "awaiting_model_assets",
        "event_id": metadata["event_id"],
        "task": "satellite_to_radar_synthesis",
        "required_assets": [
            "卫星转雷达模型代码",
            "训练好的checkpoint",
            "所需SEVIR卫星通道及配对VIL数据",
            "预处理配置与示例推理命令",
        ],
        "integration_contract": "synthesize(satellite_frames) -> vil_or_radar_frames",
        "note": "当前本地样本只有VIL通道，因此不能执行卫星转雷达推理。",
    }


def run_simulator(event_id: str, simulator_name: str = "jcm") -> dict[str, Any]:
    _, metadata = _event(event_id)
    return {
        "status": "awaiting_simulator_assets",
        "event_id": metadata["event_id"],
        "simulator_name": simulator_name,
        "required_assets": ["模拟器代码", "运行环境", "初始场数据", "配置与调用示例"],
        "integration_contract": "simulate(initial_state, duration, parameters) -> fields",
        "note": "已预留Agent工具接口，但尚未接入真实模拟器。JCM不是CASCast的依赖项。",
    }


def runtime_status() -> dict[str, Any]:
    manifest: dict[str, Any] = {"samples": 0, "events": []}
    try:
        _, metadata = _load_cascast_prediction()
        manifest = _ensemble_manifest()
        sample = {
            "available": True,
            "event_id": metadata["event_id"],
            "kind": "official_cascast_test_sample",
            "available_event_count": int(manifest["samples"]),
            "event_ids": [event["event_id"] for event in manifest["events"]],
        }
    except RuntimeError as error:
        sample = {"available": False, "error": str(error)}
    cascast_code = (CASCAST_ROOT / "networks" / "casformer.py").exists()
    cascast_checkpoint = (
        CASCAST_ROOT
        / "experiments"
        / "cascast_diffusion"
        / "world_size1-ckpt"
        / "checkpoint_best.pth"
    ).exists()
    autoencoder_checkpoint = (CASCAST_ROOT / "ckpts" / "autoencoder" / "ckpt.pth").exists()
    earthformer_latent = _earthformer_latent_status()
    latent_data = earthformer_latent["available"]
    prediction_output = CASCAST_ENSEMBLE_MANIFEST.exists()
    cascast_assets_ready = bool(
        cascast_code and cascast_checkpoint and autoencoder_checkpoint and latent_data
    )
    event_latent_available = True
    missing_assets = []
    if not cascast_code:
        missing_assets.append("代码")
    if not cascast_checkpoint:
        missing_assets.append("CasFormer checkpoint")
    if not autoencoder_checkpoint:
        missing_assets.append("Autoencoder checkpoint")
    if not latent_data:
        missing_assets.append("latent data")
    if not prediction_output:
        missing_assets.append("CasCast多事件ensemble输出")
    return {
        "sevir_sample": sample,
        "catalog": {
            "available": CATALOG_CSV.exists(),
            "vil_record_count": len(_catalog_vil_records()) if CATALOG_CSV.exists() else 0,
            "tools": [
                "exact_event_metadata",
                "data_indexer",
                "local_hdf5_loader",
                "historical_statistics",
                "projected_pixel_geolocator",
                "noaa_storm_reference",
                "history_comparison",
            ],
        },
        "dataset_evaluation": {
            "available": CASCAST_EVALUATION_REPORT.exists(),
            "event_count": int(manifest["samples"]) if prediction_output else 0,
            "ground_truth_available": bool(manifest.get("ground_truth_available", False)) if prediction_output else False,
            "report_file": str(CASCAST_EVALUATION_REPORT.relative_to(ROOT)),
        },
        "persistence_baseline": {"available": True},
        "cascast": {
            "available": bool(prediction_output),
            "assets_ready": cascast_assets_ready,
            "inference_ready": bool(cascast_assets_ready),
            "code_available": cascast_code,
            "checkpoint_available": cascast_checkpoint,
            "autoencoder_checkpoint_available": autoencoder_checkpoint,
            "latent_data_available": latent_data,
            "event_latent_available": event_latent_available,
            "prediction_output_available": prediction_output,
            "display_mode": "saved_verified_ensemble_package",
            "live_inference_for_selected_event": False,
            "ensemble_count": 10 if prediction_output else 0,
            "available_event_count": int(manifest["samples"]) if prediction_output else 0,
            "event_ids": [event["event_id"] for event in manifest["events"]] if prediction_output else [],
            "earthformer_latent": earthformer_latent,
            "reason": (
                f"仍缺少{'、'.join(missing_assets)}"
                if missing_assets
                else f"10个事件的10路CasCast ensemble已接入Demo"
            ),
        },
        "satellite_to_radar": {"available": False, "reason": "等待模型材料和卫星通道数据"},
        "simulator": {"available": False, "reason": "只预留接口，尚未指定具体模拟器"},
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sevir_event_metadata",
            "description": "按SEVIR事件ID精确获取一条VIL记录，包括严格时间轴、投影、图像尺寸和NOAA关联编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "file_name": {"type": "string"},
                    "file_index": {"type": "integer"},
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_sevir_vil_event",
            "description": "从本地官方SEVIR HDF5读取指定事件的真实VIL像素，并返回逐帧真实时间和统计摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "start_frame": {"type": "integer", "default": 0},
                    "end_frame": {"type": "integer", "default": 49},
                    "file_name": {"type": "string"},
                    "file_index": {"type": "integer"},
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sevir_catalog",
            "description": "在SEVIR CATALOG中按事件ID、年份、月份或storm type检索VIL记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "year": {"type": "integer"},
                    "month": {"type": "integer"},
                    "event_type": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 10}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "historical_sevir_statistics",
            "description": "统计SEVIR VIL历史metadata：年份、月份、storm type分布、catalog data_max分位数和强度排序。",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "event_type": {"type": "string", "default": ""}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geolocate_sevir_event",
            "description": "查询SEVIR VIL事件覆盖框；给出row和column时，使用CATALOG投影完成像素中心到经纬度的转换。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "row": {"type": "integer"},
                    "column": {"type": "integer"},
                },
                "required": ["event_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_noaa_storm_reference",
            "description": "查询SEVIR事件是否关联NOAA Storm Events，并返回episode ID、event ID和官方详情链接。",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_event_with_history",
            "description": "将指定SEVIR VIL事件的catalog data_max与同类历史记录比较并计算经验分位。",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_uploaded_vil",
            "description": "分析用户刚上传的单张SEVIR VIL图片，计算强回波统计并标出候选hazard区域；不执行未来预测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "upload_id": {"type": "string"},
                    "threshold": {"type": "number"},
                    "minimum_region_pixels": {"type": "integer", "default": 16}
                },
                "required": ["upload_id", "threshold"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_summary",
            "description": "查询本地真实SEVIR VIL样本的帧数、形状和统计量。",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_uploaded_vil_sequence",
            "description": "分析用户上传的13帧连续SEVIR VIL历史序列：逐帧强回波变化、storm-object轨迹和Persistence未来60分钟基线。不得称为CasCast预测。",
            "parameters": {
                "type": "object",
                "properties": {
                    "upload_id": {"type": "string"},
                    "threshold": {"type": "number"},
                    "minimum_region_pixels": {"type": "integer", "default": 16}
                },
                "required": ["upload_id", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_threshold",
            "description": "对真实SEVIR VIL样本执行强回波阈值统计。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "threshold": {"type": "number"},
                },
                "required": ["event_id", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_hazard",
            "description": "根据观测VIL阈值和受影响面积比例筛查强回波持续时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "threshold": {"type": "number"},
                    "min_area_ratio": {"type": "number", "default": 0.001},
                },
                "required": ["event_id", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ensemble_uncertainty",
            "description": "仅用于用户明确询问ensemble不确定性、成员分歧、超阈值概率或storm-object离散度。返回10路成员差异及预测帧；普通未来VIL预测不要使用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "threshold": {"type": "number"},
                    "minimum_region_pixels": {"type": "integer", "default": 16},
                },
                "required": ["event_id", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_cascast_dataset",
            "description": "评价当前全部10个CasCast测试事件：多阈值CSI/POD/FAR、逐时效曲线、Persistence对比、ensemble可靠性和storm-object时序验证。用户询问整体表现、测试集评价、模型对比或不确定性是否可信时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_threshold": {"type": "number", "default": 133}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_nowcast",
            "description": "用于普通临近预报：读取未来60分钟逐帧VIL数值和完整预测图。用户只问未来VIL、逐帧图像或变化趋势时选择本工具；不要用于H1/H2/H3、危险区域位置、面积或持续时间分析。Persistence使用相同13帧历史输入。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": ["persistence_baseline", "cascast"],
                    },
                    "input_frames": {"type": "integer", "default": 13},
                    "forecast_frames": {"type": "integer", "default": 12},
                    "threshold": {"type": "number"},
                },
                "required": ["event_id", "model_name", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_hazards",
            "description": "仅用于用户明确询问预测危险区域、H1/H2/H3、空间位置、影响面积、开始时间或持续时间。内部直接读取预测并返回12帧摘要，无需先调用run_nowcast；普通未来VIL预测不要使用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "model_name": {
                        "type": "string",
                        "enum": ["persistence_baseline", "cascast"],
                    },
                    "lead_minutes": {
                        "type": "integer",
                        "enum": [60, 120, 180],
                        "default": 60,
                    },
                    "threshold": {"type": "number"},
                    "minimum_region_pixels": {"type": "integer", "default": 16},
                },
                "required": ["event_id", "model_name", "threshold"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_satellite_to_radar",
            "description": "运行SEVIR卫星转雷达合成任务；当前用于报告待接入材料。",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulator",
            "description": "气象模拟器预留接口；报告真实模拟器接入状态，不生成虚假结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "simulator_name": {"type": "string", "default": "jcm"},
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "get_sevir_event_metadata": get_sevir_event_metadata,
    "load_sevir_vil_event": load_sevir_vil_event,
    "search_sevir_catalog": search_sevir_catalog,
    "historical_sevir_statistics": historical_sevir_statistics,
    "geolocate_sevir_event": geolocate_sevir_event,
    "get_noaa_storm_reference": get_noaa_storm_reference,
    "compare_event_with_history": compare_event_with_history,
    "analyze_uploaded_vil": analyze_uploaded_vil,
    "analyze_uploaded_vil_sequence": analyze_uploaded_vil_sequence,
    "get_event_summary": get_event_summary,
    "analyze_threshold": analyze_threshold,
    "analyze_hazard": analyze_hazard,
    "analyze_ensemble_uncertainty": analyze_ensemble_uncertainty,
    "evaluate_cascast_dataset": evaluate_cascast_dataset,
    "run_nowcast": run_nowcast,
    "forecast_hazards": forecast_hazards,
    "run_satellite_to_radar": run_satellite_to_radar,
    "run_simulator": run_simulator,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ValueError(f"工具不在允许列表中：{name}")
    return function(**arguments)
