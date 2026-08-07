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
