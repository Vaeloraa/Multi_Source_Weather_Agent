"""SEVIR-specific identity matching for request-bound uploads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import h5py
import numpy as np

from ...config import ROOT
from ...request.uploads import _load_uploaded_vil, _load_uploaded_vil_sequence, load_uploaded_data
from .repository import resolve_sevir_subject_key


_HISTORY_SEGMENT_STARTS = (0, 12, 24)


_INDEX_PATH = ROOT / "data" / "uploads" / "sevir_vil_history_sha256.json"


def load_uploaded_dataset(description: dict[str, Any]) -> dict[str, Any]:
    """Expose uploaded VIL pixels with the numeric conventions owned by this tool."""
    upload_id = description["upload_id"]
    pixels, metadata = load_uploaded_data(upload_id)
    values = np.ascontiguousarray(pixels, dtype=np.float32)
    values.setflags(write=False)
    sequence = metadata["kind"] == "sequence"
    return {
        "source": "upload", "role": "local_observation", "dataset_id": f"upload:{upload_id}",
        "upload_id": upload_id, "metadata": dict(metadata), "kind": metadata["kind"],
        "shape": list(values.shape), "dimensions": ["time", "row", "column"] if sequence else ["row", "column"],
        "dtype": str(values.dtype), "source_dtype": str(pixels.dtype), "missing_value_code": 255,
        "relative_minutes": list(metadata.get("history_minutes", [])), "status": "available",
        "capabilities": ["encoded_vil", "local_observation"] + (["temporal_sequence"] if sequence else []),
        "raw_data": values, "values": values,
    }


def _sources() -> list[Path]:
    root = ROOT / "data" / "sevir_complete" / "data" / "vil"
    return sorted(path.resolve() for path in root.rglob("*.h5") if path.is_file()) if root.is_dir() else []


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.uint8).tobytes()).hexdigest()


def build_hdf5_history_index(force: bool = False) -> dict[str, Any]:
    """Build the exact-pixel lookup index owned by the SEVIR capability."""
    sources = _sources()
    signatures = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sources]
    if _INDEX_PATH.is_file() and not force:
        try:
            existing = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            if existing.get("version") == 2 and existing.get("sources") == signatures:
                return existing
        except (OSError, ValueError, TypeError):
            pass
    histories: dict[str, list[dict[str, Any]]] = {}
    frames: dict[str, list[dict[str, Any]]] = {}
    count = 0
    for path in sources:
        with h5py.File(path, "r") as handle:
            if "vil" not in handle or "id" not in handle or handle["vil"].shape[-1] < 49:
                continue
            for row in range(int(handle["vil"].shape[0])):
                cube = np.asarray(handle["vil"][row], dtype=np.uint8)
                stored_id = handle["id"][row]
                event_id = stored_id.decode("ascii") if isinstance(stored_id, bytes) else str(stored_id)
                base = {"event_id": event_id, "h5_file": path.name, "file_index": row,
                        "hdf5_row": row, "source_hdf5": str(path)}
                for segment, start in enumerate(_HISTORY_SEGMENT_STARTS):
                    history = np.moveaxis(cube[:, :, start:start + 13], -1, 0)
                    histories.setdefault(_digest(history), []).append({**base, "segment": segment})
                frames.setdefault(_digest(cube[:, :, 24]), []).append(base)
                count += 1
    payload = {"version": 2, "sources": signatures, "indexed_rows": count,
               "history_13_frame_sha256": histories, "frame_24_sha256": frames}
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _INDEX_PATH.with_name(f".{_INDEX_PATH.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(_INDEX_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _verified(values: np.ndarray, key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = build_hdf5_history_index()
    candidates = index.get(key, {}).get(_digest(values), [])
    matches = []
    for candidate in candidates:
        with h5py.File(candidate["source_hdf5"], "r") as handle:
            cube = handle["vil"]
            row = int(candidate["hdf5_row"])
            if key == "history_13_frame_sha256":
                start = _HISTORY_SEGMENT_STARTS[int(candidate["segment"])]
                stored = np.moveaxis(np.asarray(cube[row, :, :, start:start + 13], dtype=np.uint8), -1, 0)
            else:
                stored = np.asarray(cube[row, :, :, 24], dtype=np.uint8)
        if np.array_equal(np.asarray(values, dtype=np.uint8), stored):
            matches.append(dict(candidate))
    return matches, index


def match_uploaded_single_frame_hdf5(upload_id: str) -> dict[str, Any]:
    frame, _ = _load_uploaded_vil(upload_id)
    matches, index = _verified(frame, "frame_24_sha256")
    if len(matches) == 1:
        return {"status": "matched", **matches[0], "method": "sha256_frame24_then_exact_pixels",
                "confidence": 1.0, "hdf5_frame_index": 24,
                "indexed_rows": int(index.get("indexed_rows", 0))}
    return {"status": "ambiguous" if matches else "not_found", "event_id": None,
            "matches": matches, "method": "sha256_frame24_then_exact_pixels",
            "confidence": None if matches else 0.0,
            "indexed_rows": int(index.get("indexed_rows", 0))}


def match_uploaded_sequence_hdf5(upload_id: str) -> dict[str, Any]:
    sequence, _ = _load_uploaded_vil_sequence(upload_id)
    matches, index = _verified(sequence, "history_13_frame_sha256")
    if len(matches) == 1:
        match = matches[0]
        start = _HISTORY_SEGMENT_STARTS[int(match["segment"])]
        return {"status": "matched", **match, "method": "sha256_history_then_exact_pixels",
                "confidence": 1.0, "history_frame_indices": list(range(start, start + 13)),
                "history_minutes": list(range(-60, 5, 5)),
                "indexed_rows": int(index.get("indexed_rows", 0))}
    return {"status": "ambiguous" if matches else "not_found", "event_id": None,
            "matches": matches, "method": "sha256_history_then_exact_pixels",
            "confidence": None if matches else 0.0,
            "indexed_rows": int(index.get("indexed_rows", 0))}


def match_uploaded_event(upload_id: str) -> dict[str, Any]:
    """Resolve an upload only through explicit metadata or exact SEVIR pixels."""
    try:
        pixels, metadata = _load_uploaded_vil(upload_id)
        kind = "single_frame"
    except ValueError:
        pixels, metadata = _load_uploaded_vil_sequence(upload_id)
        kind = "sequence"
    supplied = metadata.get("event_match")
    if isinstance(supplied, dict) and supplied.get("status") == "matched":
        event_match = dict(supplied)
    else:
        match = (match_uploaded_sequence_hdf5(upload_id) if kind == "sequence"
                 else match_uploaded_single_frame_hdf5(upload_id))
        event_match = dict(match)
        if match.get("status") == "matched":
            event_match["subject_key"] = resolve_sevir_subject_key(str(match["event_id"]))
    del pixels
    return {"status": "ok", "task": "uploaded_event_match", "upload_id": upload_id,
            "kind": kind, "event_match": event_match}
