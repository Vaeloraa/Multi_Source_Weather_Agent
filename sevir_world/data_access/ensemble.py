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
    return {
        "available": bool(valid_files),
        "root": str(EARTHFORMER_LATENT_ROOT),
        "valid_file_count": len(valid_files),
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "expected_shape": [12, 4, 48, 48],
        "event_compatible": None,
        "event_note": "Latent files are model assets; no event identity is inferred from a filename.",
    }


@lru_cache(maxsize=1)
def _ensemble_manifest() -> dict[str, Any]:
    if not CASCAST_ENSEMBLE_MANIFEST.exists():
        raise RuntimeError("未找到CasCast多事件ensemble缓存，请先运行build_ensemble_cache.py。")
    return json.loads(CASCAST_ENSEMBLE_MANIFEST.read_text(encoding="utf-8"))


def _catalog_vil_row(event_id: str) -> dict[str, str]:
    """Return the VIL catalog row used to georeference an external cache."""

    if not CATALOG_CSV.is_file():
        return {}
    with CATALOG_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("id") == event_id and row.get("img_type") == "vil":
                return dict(row)
    return {}


@lru_cache(maxsize=16)
def _live_ensemble_event(event_id: str) -> dict[str, Any] | None:
    """Discover a persisted same-event CasCast package outside the legacy manifest.

    Live/prepared inference packages used to be readable only by special demo
    helpers.  Registering them here makes the ordinary tools and fallback path
    see exactly the same cache instead of reporting that the event is missing.
    """

    requested = str(event_id).strip().upper()
    if not requested or not CASCAST_LIVE_CACHE.is_dir():
        return None
    prefix = f"{requested}_s"
    candidates = sorted(
        path
        for path in CASCAST_LIVE_CACHE.glob("*.npz")
        if path.stem.startswith(prefix) and "_roll" not in path.stem
    )
    if not candidates:
        return None
    cache_path = candidates[0]
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            history_frames = int(payload["history"].shape[0])
            member_shape = payload["ensemble_members"].shape
            forecast_frames = int(member_shape[1])
            ensemble_count = int(member_shape[0])
            file_index = int(payload["hdf5_row"].item()) if "hdf5_row" in payload.files else 0
            segment = int(payload["segment"].item()) if "segment" in payload.files else 0
            ground_truth_frames = int(payload["ground_truth"].shape[0])
    except (OSError, ValueError, KeyError, IndexError):
        return None
    row = _catalog_vil_row(requested)

    def number(name: str) -> float | None:
        try:
            value = row.get(name, "")
            return float(value) if value != "" else None
        except (TypeError, ValueError):
            return None

    source_name = Path(row.get("file_name", "")).name
    return {
        "sample_name": f"vil-{source_name or requested}-{file_index}-{segment}.npy",
        "source_h5_name": source_name,
        "file_index": file_index,
        "segment": segment,
        "id": requested,
        "time_utc": row.get("time_utc"),
        "event_type": row.get("event_type") or None,
        "llcrnrlat": number("llcrnrlat"),
        "llcrnrlon": number("llcrnrlon"),
        "urcrnrlat": number("urcrnrlat"),
        "urcrnrlon": number("urcrnrlon"),
        "data_min": number("data_min"),
        "data_max": number("data_max"),
        "event_id": requested,
        "sample_index": file_index,
        "sample_position": segment,
        "cache_file": cache_path.name,
        "cache_root": "live_cascast_cache",
        "ensemble_count": ensemble_count,
        "history_frames": history_frames,
        "forecast_frames": forecast_frames,
        "ground_truth_frames": ground_truth_frames,
    }


def _ensemble_event(event_id: str) -> dict[str, Any]:
    manifest = _ensemble_manifest()
    requested = str(event_id).strip().upper()
    if not requested:
        raise ValueError("event_id is required; no default CasCast subject is selected")
    for event in manifest["events"]:
        aliases = {
            str(event["event_id"]),
            str(event["sample_index"]),
            f"SEVIR_TEST_{event['sample_index']}",
        }
        if requested in aliases:
            return event
    live_event = _live_ensemble_event(requested)
    if live_event is not None:
        return live_event
    live_ids = {
        path.name.split("_s", 1)[0]
        for path in CASCAST_LIVE_CACHE.glob("*_s*_ens*.npz")
        if "_roll" not in path.stem
    } if CASCAST_LIVE_CACHE.is_dir() else set()
    available = "、".join(
        [*(str(event["event_id"]) for event in manifest["events"]), *sorted(live_ids)]
    )
    raise ValueError(f"当前CasCast结果不含事件{requested}；可选事件：{available}")


@lru_cache(maxsize=16)
def _load_cascast_ensemble(event_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    event = _ensemble_event(event_id)
    cache_root = (
        CASCAST_LIVE_CACHE
        if event.get("cache_root") == "live_cascast_cache"
        else CASCAST_ENSEMBLE_CACHE
    )
    cache_path = cache_root / str(event["cache_file"])
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
    member_shape = arrays["ensemble_members"].shape
    if (
        arrays["ensemble_members"].ndim != 4
        or member_shape[0] < 1
        or member_shape[1:] != (12, 384, 384)
    ):
        raise RuntimeError(
            "CasCast ensemble必须是[member, 12, 384, 384]且至少包含一个成员："
            f"{member_shape}"
        )
    expected_field_shape = member_shape[1:]
    for name in ("ground_truth", "ensemble_mean", "pixel_std", "q10", "q90"):
        if arrays[name].shape != expected_field_shape:
            raise RuntimeError(
                f"CasCast {name}形状应为{expected_field_shape}，实际为{arrays[name].shape}"
            )
    if arrays["member_seed"].shape[0] != member_shape[0]:
        raise RuntimeError("CasCast member_seed数量与ensemble成员数不一致")
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


@lru_cache(maxsize=8)
def _load_cascast_rolling_ensemble(
    event_id: str,
    segment_count: int = 3,
    ensemble_members: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a rolling 2–3 hour horizon from consecutive cached 60-minute windows."""
    if segment_count not in {2, 3}:
        raise ValueError("segment_count must be 2 or 3")
    base = _ensemble_event(event_id)
    events = _ensemble_manifest().get("events", [])
    base_position = next(index for index, item in enumerate(events) if item["event_id"] == base["event_id"])
    selected = [events[(base_position + offset) % len(events)] for offset in range(segment_count)]
    loaded = [_load_cascast_ensemble(str(item["event_id"])) for item in selected]
    first_arrays, first_meta = loaded[0]
    available_members = int(first_arrays["ensemble_members"].shape[0])
    count = available_members if ensemble_members is None else int(ensemble_members)
    if not 1 <= count <= available_members:
        raise ValueError(f"ensemble_members must be between 1 and {available_members}")
    arrays = {
        "history": first_arrays["history"],
        "ground_truth": np.concatenate([item[0]["ground_truth"] for item in loaded], axis=0),
        "ensemble_members": np.concatenate([item[0]["ensemble_members"][:count] for item in loaded], axis=1),
        "ensemble_mean": np.concatenate([item[0]["ensemble_mean"] for item in loaded], axis=0),
        "pixel_std": np.concatenate([item[0]["pixel_std"] for item in loaded], axis=0),
        "q10": np.concatenate([item[0]["q10"] for item in loaded], axis=0),
        "q90": np.concatenate([item[0]["q90"] for item in loaded], axis=0),
        "member_seed": first_arrays["member_seed"][:count],
    }
    metadata = {
        **first_meta,
        "rolling": True,
        "rolling_segment_count": segment_count,
        "rolling_source_event_ids": [item[1]["event_id"] for item in loaded],
        "ensemble_count": count,
        "forecast_minutes": list(range(5, segment_count * 60 + 1, 5)),
    }
    return arrays, metadata




def _load_cascast_prediction(event_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, metadata = _load_cascast_ensemble(event_id)
    prediction = arrays["ensemble_mean"].transpose(1, 2, 0)
    return prediction, metadata


def _vil_display_values(values: np.ndarray) -> np.ndarray:
    """Return a valid SEVIR display/engineering view without mutating raw output."""
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 254.0)


def _cascast_source(event_id: str) -> dict[str, Any]:
    event = _ensemble_event(event_id)
    cache_root = (
        CASCAST_LIVE_CACHE
        if event.get("cache_root") == "live_cascast_cache"
        else CASCAST_ENSEMBLE_CACHE
    )
    is_live_cache = cache_root == CASCAST_LIVE_CACHE
    return {
        "kind": "cascast_persisted_event_prediction" if is_live_cache else "cascast_official_test_prediction",
        "event_id": event["event_id"],
        "sample_name": event["sample_name"],
        "prediction_file": str((cache_root / event["cache_file"]).relative_to(ROOT)),
        "metadata_file": str(
            (CATALOG_CSV if is_live_cache else CASCAST_ENSEMBLE_MANIFEST).relative_to(ROOT)
        ),
        "note": "CasCast持久化真实推理结果：13帧历史观测、10路ensemble、ensemble mean与12帧ground truth。",
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
    if event_id != actual_id:
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
    if model_name == "cascast_live":
        from .live_inference import load_live_cascast_ensemble

        arrays, metadata = load_live_cascast_ensemble(event_id)
    else:
        arrays, metadata = _load_cascast_ensemble(event_id)
    history = arrays["history"].astype(np.float32)
    if model_name in {"cascast", "cascast_live"}:
        forecast = arrays["ensemble_mean"].astype(np.float32)
    elif model_name == "persistence_baseline":
        forecast = np.repeat(history[-1][None, :, :], 12, axis=0)
    else:
        raise ValueError(f"不支持的预测模型：{model_name}")
    selected_lead = forecast.shape[0] - 1 if lead_index is None else int(lead_index)
    if not 0 <= selected_lead < forecast.shape[0]:
        raise ValueError(f"预测时效序号超出范围：{selected_lead}")
    lead_slice = slice(0, selected_lead + 1)

    if series == "history":
        frames = history
    elif series == "forecast":
        frames = forecast
    elif series == "forecast_overview":
        return forecast[lead_slice].max(axis=0), metadata
    elif series == "ground_truth":
        truth = arrays["ground_truth"]
        if not 0 <= frame_index < truth.shape[0]:
            raise ValueError(f"Ground truth帧序号超出范围：{frame_index}")
        return truth[frame_index].astype(np.float32), metadata
    elif series == "uncertainty_mean":
        return arrays["ensemble_mean"][selected_lead].astype(np.float32), metadata
    elif series == "uncertainty_std":
        return arrays["pixel_std"][lead_slice].max(axis=0).astype(np.float32), metadata
    elif series == "uncertainty_probability":
        probability = (
            arrays["ensemble_members"].astype(np.float32) >= threshold
        ).mean(axis=0)
        return probability[lead_slice].max(axis=0).astype(np.float32), metadata
    elif series == "hazard_probability_any":
        probability = (
            arrays["ensemble_members"].astype(np.float32)[:, lead_slice] >= threshold
        ).any(axis=1).mean(axis=0)
        return probability.astype(np.float32), metadata
    elif series == "hazard_duration_mean":
        duration = (
            arrays["ensemble_members"].astype(np.float32)[:, lead_slice] >= threshold
        ).sum(axis=1).mean(axis=0) * 5.0
        return duration.astype(np.float32), metadata
    elif series in {"ensemble_member_final", "ensemble_member"}:
        members = arrays["ensemble_members"]
        if not 0 <= frame_index < members.shape[0]:
            raise ValueError(f"Ensemble成员序号超出范围：{frame_index}")
        return members[frame_index, selected_lead].astype(np.float32), metadata
    else:
        raise ValueError("不支持的CasCast图像序列。")
    if not 0 <= frame_index < frames.shape[0]:
        raise ValueError(f"帧序号超出范围：{frame_index}")
    return frames[frame_index], metadata
