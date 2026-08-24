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

from .config import *
from .catalog import _catalog_vil_event, _path_label
from .ensemble import _preview
from ..domain.cascast_alignment import SEGMENT_STARTS, build_validation_manifest

import h5py
import numpy as np
from pyproj import CRS, Transformer


_SEVIR_EVENT_ID_RE = re.compile(r"(?i)(?<![A-Z0-9])R\d{14}(?!\d)")
_HDF5_UPLOAD_EXAMPLE_MANIFEST = (
    ROOT / "data" / "examples" / "uploads" / "hdf5_sequence_manifest.json"
)
_HDF5_HISTORY_INDEX = ROOT / "data" / "uploads" / "sevir_vil_history_sha256.json"


def _available_vil_hdf5_files() -> list[Path]:
    root = ROOT / "data" / "sevir_complete" / "data" / "vil"
    return sorted(path.resolve() for path in root.rglob("*.h5") if path.is_file()) if root.is_dir() else []


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.uint8)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def build_hdf5_history_index(force: bool = False) -> dict[str, Any]:
    """Index every locally available SEVIR VIL row by exact input pixels.

    The index is only a lookup accelerator. A candidate is always re-read and
    checked pixel-for-pixel before an upload is associated with an event.
    """
    sources = _available_vil_hdf5_files()
    signatures = [
        {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sources
    ]
    if _HDF5_HISTORY_INDEX.is_file() and not force:
        try:
            existing = json.loads(_HDF5_HISTORY_INDEX.read_text(encoding="utf-8"))
            if existing.get("sources") == signatures and existing.get("version") == 2:
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    history: dict[str, list[dict[str, Any]]] = {}
    frame24: dict[str, list[dict[str, Any]]] = {}
    row_count = 0
    for path in sources:
        with h5py.File(path, "r") as handle:
            if "vil" not in handle or "id" not in handle:
                continue
            vil = handle["vil"]
            ids = handle["id"]
            if vil.ndim != 4 or vil.shape[-1] < 49:
                continue
            for start in range(0, int(vil.shape[0]), 8):
                stop = min(start + 8, int(vil.shape[0]))
                block = np.asarray(vil[start:stop], dtype=np.uint8)
                for offset in range(stop - start):
                    row = start + offset
                    stored_id = ids[row]
                    event_id = stored_id.decode("ascii") if isinstance(stored_id, bytes) else str(stored_id)
                    base_candidate = {
                        "event_id": event_id,
                        "h5_file": path.name,
                        "file_index": row,
                        "hdf5_row": row,
                        "source_hdf5": str(path),
                    }
                    for segment, frame_start in enumerate(SEGMENT_STARTS):
                        sequence = np.moveaxis(
                            block[offset, :, :, frame_start : frame_start + 13], -1, 0
                        )
                        candidate = {**base_candidate, "segment": segment}
                        history.setdefault(_sha256_array(sequence), []).append(candidate)
                    frame24.setdefault(
                        _sha256_array(block[offset, :, :, 24]), []
                    ).append(base_candidate)
                    row_count += 1
    payload = {
        "version": 2,
        "method": "sha256_lookup_then_exact_pixel_verification",
        "sources": signatures,
        "indexed_rows": row_count,
        "history_13_frame_sha256": history,
        "frame_24_sha256": frame24,
    }
    _HDF5_HISTORY_INDEX.parent.mkdir(parents=True, exist_ok=True)
    # Several isolated Agent workers can request the index concurrently.  A
    # fixed .tmp name lets one process replace a file while another process is
    # still writing it on Windows, leaving a truncated JSON index.  A unique
    # sibling plus atomic replace keeps readers on either the old or new whole
    # document.
    temporary = _HDF5_HISTORY_INDEX.with_name(
        f".{_HDF5_HISTORY_INDEX.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(_HDF5_HISTORY_INDEX)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _verified_hdf5_candidates(
    pixels: np.ndarray,
    index_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = build_hdf5_history_index()
    digest = _sha256_array(pixels)
    candidates = index.get(index_key, {}).get(digest, [])
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        path = Path(candidate["source_hdf5"])
        with h5py.File(path, "r") as handle:
            vil = handle["vil"]
            row = int(candidate["hdf5_row"])
            if index_key == "history_13_frame_sha256":
                segment = int(candidate["segment"])
                frame_start = SEGMENT_STARTS[segment]
                stored = np.moveaxis(
                    np.asarray(vil[row, :, :, frame_start : frame_start + 13], dtype=np.uint8),
                    -1,
                    0,
                )
            else:
                stored = np.asarray(vil[row, :, :, 24], dtype=np.uint8)
            if np.array_equal(np.asarray(pixels, dtype=np.uint8), stored):
                matches.append(dict(candidate))
    return matches, index


def _event_match_metadata(
    filename: str,
    supplied_event_id: str = "",
    supplied_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve only explicit upload metadata; never infer an event from pixels."""
    metadata = dict(supplied_metadata or {})
    explicit_id = str(supplied_event_id or metadata.get("event_id") or "").strip()
    filename_match = _SEVIR_EVENT_ID_RE.search(Path(filename).name)
    filename_id = filename_match.group(0).upper() if filename_match else ""
    candidate = explicit_id.upper() or filename_id
    if not candidate:
        return {
            "status": "not_provided",
            "event_id": None,
            "candidates": [],
            "confidence": None,
            "method": None,
            "note": "上传内容没有提供SEVIR event_id，未根据像素外观猜测事件。",
        }
    try:
        record = _catalog_vil_event(candidate)
    except (RuntimeError, ValueError) as error:
        return {
            "status": "not_found",
            "event_id": None,
            "requested_event_id": candidate,
            "candidates": [],
            "confidence": 0.0,
            "method": "explicit_metadata" if explicit_id else "filename",
            "note": f"提供的event_id不在本地SEVIR VIL目录中：{candidate}。",
            "error": str(error),
        }
    method = "explicit_metadata" if explicit_id else "filename"
    return {
        "status": "matched",
        "event_id": record["id"],
        "candidates": [{
            "event_id": record["id"],
            "score": 1.0,
            "match_basis": method,
        }],
        "confidence": 1.0,
        "method": method,
        "metadata_source": "SEVIR CATALOG",
        "note": "仅依据明确的event_id或文件名匹配；不是基于图像像素反推位置。",
    }


def event_match_from_metadata(
    filename: str,
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public deterministic matcher used by upload handlers and tests."""
    return _event_match_metadata(filename, supplied_event_id=event_id, supplied_metadata=metadata)


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
    # Windows environments used by the demo may deny creation of dot-prefixed
    # files in the uploads directory. Use ordinary temporary names; the final
    # replace remains atomic and the cleanup path is unchanged.
    temporary_array = UPLOAD_STORAGE_DIR / f"tmp_{upload_id}_{nonce}.npz"
    temporary_metadata = UPLOAD_STORAGE_DIR / f"tmp_{upload_id}_{nonce}.json"
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
                    "grayscale_source", "all_grayscale_source", "created_at_utc", "event_match",
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
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
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
    provided_metadata = dict(metadata or {})
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
    event_match = _event_match_metadata(
        filename, supplied_event_id=event_id, supplied_metadata=provided_metadata
    )
    metadata["event_match"] = event_match
    if len(UPLOADED_VIL) >= 16:
        UPLOADED_VIL.pop(next(iter(UPLOADED_VIL)))
    UPLOADED_VIL[upload_id] = (frame.copy(), metadata)
    try:
        _persist_uploaded_array(upload_id, frame, metadata)
    except OSError as error:
        # The hosted demo may run under an account that can read the workspace
        # but cannot create files in the checked-in uploads directory. Keep the
        # upload usable in the in-memory registry for this server session.
        metadata["persisted"] = False
        metadata["persistence_error"] = type(error).__name__
    return {
        "status": "ok",
        **metadata,
        "min_vil": int(frame.min()),
        "max_vil": int(frame.max()),
        "mean_vil": round(float(frame.mean()), 4),
        "preview_grid": _preview(frame),
        "event_match": event_match,
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


def match_uploaded_event(upload_id: str) -> dict[str, Any]:
    """Resolve an upload to a SEVIR event with deterministic evidence.

    Upload registration performs only the cheap filename/metadata check.  If
    that did not identify an event, this on-demand method hashes the canonical
    frame or 13-frame history, finds local HDF5 candidates, and then requires
    exact byte-for-byte pixel equality.  It never guesses from visual shape or
    approximate image similarity, and therefore may honestly return no match.
    """
    normalized = str(upload_id or "").strip()
    if not _valid_upload_id(normalized):
        raise ValueError("upload_id格式不正确。")
    loaded = _restore_uploaded_array(normalized, expected_kind="single_frame")
    kind = "single_frame"
    if loaded is None:
        loaded = _restore_uploaded_array(normalized, expected_kind="sequence")
        kind = "sequence"
    if loaded is None:
        raise ValueError("没有找到这次上传，无法执行事件匹配。")
    pixels, metadata = loaded
    stored_event_match = metadata.get("event_match")
    event_match = dict(stored_event_match) if isinstance(stored_event_match, dict) else stored_event_match
    if not isinstance(event_match, dict):
        filenames = metadata.get("filenames") or [metadata.get("filename", "")]
        event_match = _event_match_metadata(str(filenames[0] or ""))
    if not event_match.get("match_complete") and event_match.get("status") != "matched":
        pixel_match = (
            match_uploaded_sequence_hdf5(normalized)
            if kind == "sequence"
            else match_uploaded_single_frame_hdf5(normalized)
        )
        if pixel_match.get("status") == "matched":
            event_match = {
                "status": "matched",
                "event_id": pixel_match["event_id"],
                "candidates": [{
                    "event_id": pixel_match["event_id"],
                    "score": 1.0,
                    "match_basis": pixel_match["method"],
                }],
                "confidence": 1.0,
                "method": pixel_match["method"],
                "metadata_source": "local SEVIR HDF5 exact pixels",
                "note": "Matched by SHA-256 lookup followed by exact pixel verification; no visual similarity guess was used.",
            }
        event_match["match_complete"] = True
    # Persist every deterministic outcome (matched, not_found, or ambiguous)
    # so follow-up turns reuse the upload-time decision instead of rescanning.
    if event_match != stored_event_match:
        metadata["event_match"] = event_match
        _persist_uploaded_array(normalized, pixels, metadata)
    return {
        "status": "ok",
        "task": "uploaded_event_match",
        "upload_id": normalized,
        "kind": kind,
        "event_match": event_match,
        "interpretation_limit": (
            "匹配使用上传时提供的event_id/SEVIR文件名，或本地HDF5的哈希候选加逐像素完全一致验证；"
            "不会根据图像外观做近似猜测，无法唯一验证时不会给出地理位置。"
        ),
    }


def register_uploaded_vil_sequence(
    frames: list[dict[str, Any]],
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    provided_metadata = dict(metadata or {})
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
    event_match = _event_match_metadata(
        filenames[0] if filenames else "",
        supplied_event_id=event_id,
        supplied_metadata=provided_metadata,
    )
    metadata["event_match"] = event_match
    if len(UPLOADED_VIL_SEQUENCES) >= 8:
        UPLOADED_VIL_SEQUENCES.pop(next(iter(UPLOADED_VIL_SEQUENCES)))
    UPLOADED_VIL_SEQUENCES[upload_id] = (sequence, metadata)
    try:
        _persist_uploaded_array(upload_id, sequence, metadata)
    except OSError as error:
        metadata["persisted"] = False
        metadata["persistence_error"] = type(error).__name__
    return {
        "status": "ok",
        **metadata,
        "frame_count": 13,
        "min_vil": int(sequence.min()),
        "max_vil": int(sequence.max()),
        "mean_vil": round(float(sequence.mean()), 4),
        "event_match": event_match,
        "note": (
            "已按文件名排序读取13帧；原始灰度序列可用于本地EarthFormer、CasCast和VAE现场推理。"
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


def match_uploaded_single_frame_hdf5(upload_id: str) -> dict[str, Any]:
    """Match a single upload only to the canonical HDF5 t=0 frame (index 24)."""
    frame, _ = _load_uploaded_vil(upload_id)
    matches, index = _verified_hdf5_candidates(frame, "frame_24_sha256")
    if len(matches) == 1:
        return {
            "status": "matched",
            **matches[0],
            "method": "sha256_frame24_then_exact_pixels",
            "confidence": 1.0,
            "hdf5_frame_index": 24,
            "indexed_rows": int(index.get("indexed_rows", 0)),
        }
    return {
        "status": "ambiguous" if matches else "not_found",
        "event_id": None,
        "matches": matches,
        "method": "sha256_frame24_then_exact_pixels",
        "confidence": None if matches else 0.0,
        "indexed_rows": int(index.get("indexed_rows", 0)),
        "reason": (
            "more than one HDF5 event has the same canonical t=0 frame"
            if matches else "the uploaded image is not byte-identical to any indexed HDF5 frame 24"
        ),
    }


def match_uploaded_sequence_hdf5(upload_id: str) -> dict[str, Any]:
    """Identify an uploaded 13-frame history by exact HDF5 pixel identity.

    Every row in every locally available VIL HDF5 is indexed by SHA-256. A
    candidate is accepted only when all 13 frames are byte-for-byte equal to
    one of the three CasCast input windows (0..12, 12..24, or 24..36). This is
    an identity and segment-position check, not a visual similarity guess.
    """
    sequence, _ = _load_uploaded_vil_sequence(upload_id)
    matches, index = _verified_hdf5_candidates(sequence, "history_13_frame_sha256")
    if len(matches) == 1:
        match = matches[0]
        frame_start = SEGMENT_STARTS[int(match["segment"])]
        return {
            "status": "matched",
            **match,
            "method": "sha256_history_then_exact_pixels",
            "confidence": 1.0,
            "history_frame_indices": list(range(frame_start, frame_start + 13)),
            "history_minutes": list(range(-60, 5, 5)),
            "indexed_rows": int(index.get("indexed_rows", 0)),
            "candidate_rows_checked": len(index.get("history_13_frame_sha256", {}).get(_sha256_array(sequence), [])),
        }
    return {
        "status": "ambiguous" if matches else "not_found",
        "event_id": None,
        "matches": matches,
        "method": "sha256_history_then_exact_pixels",
        "confidence": None if matches else 0.0,
        "indexed_rows": int(index.get("indexed_rows", 0)),
        "candidate_rows_checked": len(index.get("history_13_frame_sha256", {}).get(_sha256_array(sequence), [])),
        "reason": (
            "more than one candidate has identical history pixels"
            if matches
            else "the uploaded 13-frame history is not byte-identical to a local HDF5 candidate"
        ),
    }


def load_uploaded_sequence_hdf5_ground_truth(
    upload_id: str, lead_minutes: list[int]
) -> dict[str, Any]:
    """Load held-out future frames only after exact uploaded-history matching."""
    match = match_uploaded_sequence_hdf5(upload_id)
    if match["status"] != "matched":
        return {
            "available": False,
            "match": match,
            "lead_minutes": list(lead_minutes),
            "frames": [],
            "reason": "no unique exact HDF5 history match; future truth was not loaded",
        }
    leads = list(dict.fromkeys(int(value) for value in lead_minutes))
    with h5py.File(match["source_hdf5"], "r") as handle:
        vil = handle["vil"]
        validation = build_validation_manifest(
            match,
            forecast_mode="free_running",
            requested_lead_minutes=leads,
            event_frame_count=int(vil.shape[-1]),
        )
        maximum_lead = int(validation["max_valid_lead_minutes"])
        verified_leads = validation["verified_lead_minutes"]
        input_end = validation["input_frames"][-1]
        frame_indices = [input_end + lead // 5 for lead in verified_leads]
        frames = [
            np.asarray(
                vil[int(match["hdf5_row"]), :, :, frame_index],
                dtype=np.uint8,
            ).tolist()
            for frame_index in frame_indices
        ]
    return {
        "available": bool(verified_leads),
        "event_id": match["event_id"],
        "h5_file": match.get("h5_file") or Path(match["source_hdf5"]).name,
        "file_index": match.get("file_index", match["hdf5_row"]),
        "segment": match["segment"],
        "hdf5_row": match["hdf5_row"],
        "match_method": match["method"],
        "match_confidence": match["confidence"],
        "history_frame_indices": match["history_frame_indices"],
        "lead_minutes": verified_leads,
        "requested_lead_minutes": leads,
        "unverified_lead_minutes": validation["unverified_lead_minutes"],
        "hdf5_frame_indices": frame_indices,
        "validation_manifest": validation,
        "max_valid_lead_minutes": maximum_lead,
        "shape": [384, 384],
        "dtype": "uint8",
        "missing_value": 255,
        "frames": frames,
        "source": "local_sevir_hdf5_held_out_future",
        "reason": None if verified_leads else "requested leads extend beyond the matched event",
    }


def get_uploaded_sequence_display_frame(
    upload_id: str,
    series: str,
    frame_index: int,
    model_name: str = "persistence_baseline",
    lead_index: int | None = None,
) -> np.ndarray:
    sequence, _ = _load_uploaded_vil_sequence(upload_id)
    if series == "history":
        if not 0 <= frame_index < 13:
            raise IndexError("历史帧编号超出范围。")
        return sequence[frame_index].astype(np.float32)
    if series in {"forecast", "forecast_overview", "ensemble_member", "uncertainty_mean", "uncertainty_std", "uncertainty_probability"}:
        if model_name == "cascast_live":
            from .live_inference import load_uploaded_cascast_ensemble

            arrays, _ = load_uploaded_cascast_ensemble(upload_id)
            forecast = arrays["ensemble_mean"].astype(np.float32)
            selected_lead = int(lead_index if lead_index is not None else 11)
            if not 0 <= selected_lead < forecast.shape[0]:
                raise IndexError("Forecast lead index is outside the available range")
            if series == "uncertainty_mean":
                return forecast[selected_lead]
            if series == "uncertainty_std":
                members = arrays["ensemble_members"].astype(np.float32)
                return members[:, selected_lead].std(axis=0)
            if series == "uncertainty_probability":
                members = arrays["ensemble_members"].astype(np.float32)
                return (members[:, selected_lead] >= 133.0).mean(axis=0)
            if series == "ensemble_member":
                members = arrays["ensemble_members"].astype(np.float32)
                if not 0 <= frame_index < members.shape[0]:
                    raise IndexError("Ensemble成员编号超出范围。")
                if not 0 <= selected_lead < members.shape[1]:
                    raise IndexError("Ensemble预测时效超出范围。")
                return members[frame_index, selected_lead]
            if series == "forecast_overview":
                return forecast.max(axis=0)
            if not 0 <= frame_index < 12:
                raise IndexError("Forecast frame index is outside 0 through 11")
            return forecast[frame_index]
        if not 0 <= frame_index < 12 and series == "forecast":
            raise IndexError("预测帧编号超出范围。")
        return sequence[-1].astype(np.float32)
    raise ValueError(f"不支持的上传序列图像类型：{series}")
