"""CasCast input/history and ground-truth alignment contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


EVENT_FRAME_COUNT = 49
INPUT_FRAME_COUNT = 13
FORECAST_FRAME_COUNT = 12
FRAME_INTERVAL_MINUTES = 5
SEGMENT_STARTS = (0, 12, 24)

_EVENT_ID_RE = re.compile(r"(?i)(?<![A-Z0-9])(R\d{14})(?!\d)")
_SEGMENT_RE = re.compile(r"(?i)(?:^|[_-])(?:segment|seg|s)[_-]?([012])(?:$|[_-])")
_FILE_INDEX_RE = re.compile(r"(?i)(?:^|[_-])(?:file[_-]?index|row)[_-]?(\d+)(?:$|[_-])")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer, got {value!r}") from error


def parse_sample_identity(
    sample_name: str = "", metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the fields required to align a CasCast sample to absolute frames.

    Explicit metadata takes precedence over tokens embedded in ``sample_name``.
    Missing fields remain ``None`` so callers cannot accidentally treat a guess
    as an established sample identity.
    """
    source = dict(metadata or {})
    name = Path(str(sample_name or source.get("sample_name") or "")).name
    event_match = _EVENT_ID_RE.search(name)
    segment_match = _SEGMENT_RE.search(name)
    file_index_match = _FILE_INDEX_RE.search(name)

    event_id = source.get("event_id") or source.get("id")
    if not event_id and event_match:
        event_id = event_match.group(1).upper()
    segment = _optional_int(
        source.get("segment") if source.get("segment") not in (None, "")
        else segment_match.group(1) if segment_match else None
    )
    if segment is not None and segment not in range(len(SEGMENT_STARTS)):
        raise ValueError("segment must be 0, 1, or 2")
    file_index = _optional_int(
        source.get("file_index")
        if source.get("file_index") not in (None, "")
        else source.get("hdf5_row")
        if source.get("hdf5_row") not in (None, "")
        else file_index_match.group(1) if file_index_match else None
    )
    h5_file = (
        source.get("h5_file")
        or source.get("file_name")
        or source.get("source_hdf5")
    )
    return {
        "event_id": str(event_id) if event_id not in (None, "") else None,
        "h5_file": str(h5_file) if h5_file not in (None, "") else None,
        "file_index": file_index,
        "segment": segment,
    }


def build_validation_manifest(
    identity: Mapping[str, Any],
    *,
    forecast_mode: str = "free_running",
    requested_lead_minutes: list[int] | None = None,
    event_frame_count: int = EVENT_FRAME_COUNT,
) -> dict[str, Any]:
    """Describe exact input/GT alignment and the verifiable forecast horizon."""
    parsed = parse_sample_identity(metadata=identity)
    segment = parsed["segment"]
    if segment is None:
        raise ValueError("segment is required to build a validation manifest")
    if forecast_mode not in {
        "standard_single_window", "free_running", "observation_reset"
    }:
        raise ValueError(
            "forecast_mode must be standard_single_window, free_running, or observation_reset"
        )
    frame_count = int(event_frame_count)
    if frame_count < 1:
        raise ValueError("event_frame_count must be positive")

    input_start = SEGMENT_STARTS[segment]
    input_end = input_start + INPUT_FRAME_COUNT - 1
    if input_end >= frame_count:
        raise ValueError("the input window extends beyond the event")
    available_gt = list(range(input_end + 1, frame_count))
    standard_gt = available_gt[:FORECAST_FRAME_COUNT]
    max_valid_lead = len(available_gt) * FRAME_INTERVAL_MINUTES
    available_leads = list(
        range(FRAME_INTERVAL_MINUTES, max_valid_lead + 1, FRAME_INTERVAL_MINUTES)
    )
    requested = list(dict.fromkeys(int(value) for value in (requested_lead_minutes or [])))
    invalid_grid = [value for value in requested if value < 5 or value % 5]
    if invalid_grid:
        raise ValueError("requested lead times must be positive 5-minute increments")
    verified = [value for value in requested if value <= max_valid_lead]
    unverified = [value for value in requested if value > max_valid_lead]

    return {
        **parsed,
        "input_frames": list(range(input_start, input_end + 1)),
        "standard_gt_frames": standard_gt,
        "available_gt_frames": available_gt,
        "standard_max_valid_lead_minutes": len(standard_gt) * FRAME_INTERVAL_MINUTES,
        "max_valid_lead_minutes": max_valid_lead,
        "available_lead_minutes": available_leads,
        "requested_lead_minutes": requested,
        "verified_lead_minutes": verified,
        "unverified_lead_minutes": unverified,
        "forecast_mode": forecast_mode,
        "ground_truth_complete_for_requested_horizon": not unverified,
        "frame_interval_minutes": FRAME_INTERVAL_MINUTES,
        "event_frame_count": frame_count,
    }


def load_uploaded_sequence_hdf5_ground_truth(
    upload_id: str, lead_minutes: list[int]
) -> dict[str, Any]:
    """Load held-out SEVIR frames under the CasCast validation contract."""
    from ..sevir.uploads import match_uploaded_sequence_hdf5

    match = match_uploaded_sequence_hdf5(upload_id)
    if match["status"] != "matched":
        return {
            "available": False,
            "match": match,
            "lead_minutes": [],
            "frames": [],
            "reason": "no unique exact HDF5 history match",
        }
    leads = list(dict.fromkeys(int(value) for value in lead_minutes))
    with h5py.File(match["source_hdf5"], "r") as handle:
        validation = build_validation_manifest(
            match,
            forecast_mode="free_running",
            requested_lead_minutes=leads,
            event_frame_count=int(handle["vil"].shape[-1]),
        )
        verified = validation["verified_lead_minutes"]
        input_end = validation["input_frames"][-1]
        indices = [input_end + lead // FRAME_INTERVAL_MINUTES for lead in verified]
        frames = [
            np.asarray(
                handle["vil"][int(match["hdf5_row"]), :, :, index], dtype=np.uint8
            ).tolist()
            for index in indices
        ]
    return {
        "available": bool(verified),
        "event_id": match["event_id"],
        "h5_file": match["h5_file"],
        "file_index": match["hdf5_row"],
        "segment": match["segment"],
        "lead_minutes": verified,
        "unverified_lead_minutes": validation["unverified_lead_minutes"],
        "validation_manifest": validation,
        "shape": [384, 384],
        "dtype": "uint8",
        "missing_value": 255,
        "frames": frames,
        "source": "local_sevir_hdf5_held_out_future",
    }
