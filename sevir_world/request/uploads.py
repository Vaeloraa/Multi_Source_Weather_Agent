from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..persistence.uploads import (
    UPLOADED_VIL,
    UPLOADED_VIL_SEQUENCES,
    UPLOAD_STORAGE_DIR,
    UPLOAD_STORAGE_LOCK,
)
from ..tools.sevir.repository import (
    catalog_subject_key,
    catalog_vil_event,
    portable_path_label,
)

import numpy as np


_SEVIR_EVENT_ID_RE = re.compile(r"(?i)(?<![A-Z0-9])R\d{14}(?!\d)")

def event_match_from_metadata(
    filename: str,
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve only explicit upload metadata; never infer an event from pixels."""
    metadata = metadata or {}
    explicit_id = str(event_id or metadata.get("event_id") or "").strip()
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
            "note": "No SEVIR event_id was provided. The upload remains unlinked to a catalog event.",
        }
    try:
        record = catalog_vil_event(candidate)
    except (RuntimeError, ValueError) as error:
        return {
            "status": "not_found",
            "event_id": None,
            "requested_event_id": candidate,
            "candidates": [],
            "confidence": 0.0,
            "method": "explicit_metadata" if explicit_id else "filename",
            "note": f"The event_id is not in the local SEVIR VIL catalog: {candidate}.",
            "error": str(error),
        }
    method = "explicit_metadata" if explicit_id else "filename"
    return {
        "status": "matched",
        "event_id": record["id"],
        "subject_key": catalog_subject_key(record),
        "candidates": [{
            "event_id": record["id"],
            "subject_key": catalog_subject_key(record),
            "score": 1.0,
            "match_basis": method,
        }],
        "confidence": 1.0,
        "method": method,
        "metadata_source": "SEVIR CATALOG",
        "note": "Matched using an explicit event_id or filename.",
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


def uploaded_event_match(upload_id: str) -> dict[str, Any]:
    """Read the upload-time event decision without loading pixel arrays.

    Request binding needs the cheap metadata decision during capability
    construction.  Exact pixel/HDF5 matching remains in
    :func:`match_uploaded_event` and is only performed when the Agent asks
    for that evidence explicitly.
    """

    normalized = str(upload_id or "").strip()
    _, metadata_path = _upload_storage_paths(normalized)
    metadata: dict[str, Any] | None = None
    in_memory = UPLOADED_VIL.get(normalized) or UPLOADED_VIL_SEQUENCES.get(normalized)
    if in_memory is not None:
        metadata = dict(in_memory[1])
    elif metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata is None:
        raise ValueError("Upload not found. Event binding is unavailable.")
    if metadata.get("status") == "invalid":
        return {"status": "invalid", "upload_id": normalized,
                "event_match": {}, "diagnostic": dict(metadata)}
    return {
        "status": "ok",
        "upload_id": normalized,
        "event_match": dict(metadata["event_match"]),
    }


def _register_uploaded_array(
    upload_id: str,
    pixels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    array_path, metadata_path = _upload_storage_paths(upload_id)
    UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary_array = UPLOAD_STORAGE_DIR / f"tmp_{upload_id}_{nonce}.npz"
    temporary_metadata = UPLOAD_STORAGE_DIR / f"tmp_{upload_id}_{nonce}.json"
    stored_metadata = {**metadata, "persisted": True}
    with UPLOAD_STORAGE_LOCK:
        try:
            np.savez_compressed(temporary_array, pixels=pixels)
            temporary_metadata.write_text(
                json.dumps(stored_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_array.replace(array_path)
            temporary_metadata.replace(metadata_path)
            metadata["persisted"] = True
            registry = UPLOADED_VIL if metadata["kind"] == "single_frame" else UPLOADED_VIL_SEQUENCES
            limit = 16 if metadata["kind"] == "single_frame" else 8
            if len(registry) >= limit:
                registry.pop(next(iter(registry)))
            registry[upload_id] = (pixels, metadata)
        finally:
            temporary_array.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)


def load_uploaded_data(
    upload_id: str,
    *,
    expected_kind: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and cache one upload, validating disk data at the storage boundary."""
    array_path, metadata_path = _upload_storage_paths(upload_id)
    with UPLOAD_STORAGE_LOCK:
        cached = UPLOADED_VIL.get(upload_id) or UPLOADED_VIL_SEQUENCES.get(upload_id)
        if cached is None:
            if not array_path.is_file() or not metadata_path.is_file():
                raise ValueError("The uploaded SEVIR data could not be found. Upload it again.")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = cached[1]
        kind = metadata["kind"]
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"upload kind is {kind}, expected {expected_kind}")
        if cached is None:
            with np.load(array_path, allow_pickle=False) as payload:
                pixels = payload["pixels"]
            expected_shape = (384, 384) if kind == "single_frame" else (13, 384, 384)
            if pixels.shape != expected_shape or pixels.dtype != np.uint8:
                raise ValueError(f"Persisted upload {upload_id} has invalid shape or dtype")
            cached = (pixels, metadata)
            registry = UPLOADED_VIL if kind == "single_frame" else UPLOADED_VIL_SEQUENCES
            registry[upload_id] = cached
        return cached


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
        "storage": portable_path_label(UPLOAD_STORAGE_DIR),
        "record_count": len(records),
        "records": records,
    }


def _decode_frame(
    width: int,
    height: int,
    pixels_base64: str,
) -> np.ndarray:
    if (width, height) != (384, 384):
        raise ValueError("Upload a 384 x 384 SEVIR VIL image.")
    try:
        raw = base64.b64decode(pixels_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Could not decode the image pixels.") from error
    if len(raw) != width * height:
        raise ValueError("The pixel count does not match a 384 x 384 image.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).copy()


def register_uploaded_vil(
    filename: str,
    width: int,
    height: int,
    pixels_base64: str,
    grayscale_source: bool,
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = _decode_frame(width, height, pixels_base64)
    event_match = event_match_from_metadata(filename, event_id, metadata)
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
        "event_match": event_match,
    }
    _register_uploaded_array(upload_id, frame, metadata)
    return {
        "status": "ok",
        **metadata,
        "note": (
            "Loaded with the original grayscale values."
            if grayscale_source
            else "Color image converted to grayscale for image-intensity analysis."
        ),
    }


def _load_uploaded_vil(upload_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    return load_uploaded_data(upload_id, expected_kind="single_frame")


def register_uploaded_vil_sequence(
    frames: list[dict[str, Any]],
    event_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    upload_id = uuid.uuid4().hex

    def invalid(reason: str, index: int | None = None) -> dict[str, Any]:
        diagnostic = {
            "status": "invalid",
            "upload_id": upload_id,
            "kind": "sequence",
            "frame_count_received": len(frames),
            "expected_frame_count": 13,
            "invalid_frame_index": index,
            "reason": reason,
            "filenames": [Path(str(item.get("filename", ""))).name for item in frames if isinstance(item, dict)],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _, metadata_path = _upload_storage_paths(upload_id)
        metadata_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        return diagnostic

    if len(frames) != 13:
        return invalid("A sequence requires exactly 13 grayscale SEVIR VIL images at 384 x 384 pixels.")
    decoded: list[np.ndarray] = []
    filenames: list[str] = []
    grayscale_flags: list[bool] = []
    for index, item in enumerate(frames):
        try:
            decoded.append(_decode_frame(
                int(item.get("width", 0)), int(item.get("height", 0)),
                item.get("pixels_base64", ""),
            ))
        except (TypeError, ValueError) as error:
            return invalid(f"Image {index + 1}: {error}", index)
        filenames.append(Path(str(item.get("filename", f"frame_{index + 1:02d}.png"))).name)
        grayscale_flags.append(bool(item.get("grayscale_source", False)))
    sequence = np.stack(decoded, axis=0)
    provided_metadata = dict(metadata or {})
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
    if provided_metadata.get("initialization_time_utc"):
        metadata["initialization_time_utc"] = str(
            provided_metadata["initialization_time_utc"]
        ).strip()
    event_match = event_match_from_metadata(
        filenames[0], event_id, provided_metadata,
    )
    metadata["event_match"] = event_match
    _register_uploaded_array(upload_id, sequence, metadata)
    return {
        "status": "ok",
        **metadata,
        "frame_count": 13,
        "note": (
            "Loaded 13 frames in filename order for local EarthFormer, CasCast and VAE inference."
        ),
    }


def _load_uploaded_vil_sequence(upload_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    return load_uploaded_data(upload_id, expected_kind="sequence")
