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
from .catalog import _path_label
from .ensemble import _preview

import h5py
import numpy as np
from pyproj import CRS, Transformer


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
