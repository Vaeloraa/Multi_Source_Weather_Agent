"""Multi-modal SEVIR data loading for VIS, IR069, IR107, and LGHT channels.

This module provides functions to load and process satellite imagery and lightning
data alongside the existing VIL radar data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from weather_agent.tools.sevir.repository import (
    _catalog_event_by_type,
    _catalog_subject_key,
    _full_catalog_record,
    _load_array_by_type,
)


# Channel metadata: physical interpretation and value ranges
CHANNEL_METADATA = {
    "vis": {
        "name": "Visible",
        "description": "GOES-16 C02 visible imagery (0.64 μm)",
        "units": "scaled reflectance",
        "missing_value": None,
        "typical_range": [0, 1],
        "interpretation": "Cloud boundaries, texture, anvil clouds, spatial organization",
        "limitations": "Not available at night, affected by solar elevation angle",
    },
    "ir069": {
        "name": "Water Vapor",
        "description": "GOES-16 C09 mid-level water vapor (6.9 μm)",
        "units": "brightness temperature (K)",
        "missing_value": None,
        "typical_range": [180, 280],
        "interpretation": "Mid-upper tropospheric moisture, dry air intrusion, environmental humidity structure",
        "limitations": "Not direct precipitation observation",
    },
    "ir107": {
        "name": "Infrared Window",
        "description": "GOES-16 C13 longwave infrared (10.7 μm)",
        "units": "brightness temperature (K)",
        "missing_value": None,
        "typical_range": [180, 320],
        "interpretation": "Cloud-top temperature, identifies rapid cooling as convection develops",
        "limitations": "Cannot directly infer surface hazards",
    },
    "lght": {
        "name": "Lightning",
        "description": "GLM (Geostationary Lightning Mapper) flash extent density",
        "units": "flashes",
        "missing_value": None,
        "typical_range": [0, 50],
        "interpretation": "Direct observation of electrical activity, indicates when lightning begins, intensifies, or weakens",
        "limitations": "Only indicates electrical activity, not equivalent to tornado or hail",
    },
}


def load_sevir_multimodal_event(
    event_id: str,
    channels: list[str] | None = None,
    start_frame: int = 0,
    end_frame: int = 49,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Load multi-modal SEVIR data for a single event across multiple channels.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    channels : list of str, optional
        List of channels to load. Available: 'vil', 'vis', 'ir069', 'ir107', 'lght'.
        ``lght`` is the SEVIR catalog/API code for GLM lightning data; do not
        pass ``GLM`` as a channel value.
        If None, loads all available channels for the event.
    start_frame : int, default 0
        Starting frame index (inclusive).
    end_frame : int, default 49
        Ending frame index (exclusive).
    minutes_before : int, optional
        If provided, overrides start_frame to select frames from this many minutes
        before the event anchor time.
    minutes_after : int, optional
        If provided, overrides end_frame to select frames up to this many minutes
        after the event anchor time.
    file_name : str, optional
        Source file name for disambiguation.
    file_index : int, optional
        Dataset index for disambiguation.

    Returns
    -------
    dict
        Dictionary with keys for each loaded channel, plus metadata.
        Each channel contains:
        - frames: numpy array of shape (time, height, width)
        - metadata: channel-specific information
        - frame_times_utc: list of UTC timestamps for each frame

    Supply the event identity selected for the current request. Use ``channels``
    to restrict the requested observations; no demonstration event is selected
    by this function.
    """
    if channels is None:
        channels = ["vil", "vis", "ir069", "ir107", "lght"]

    # Normalize channel names
    channels = [ch.lower().strip() for ch in channels]
    valid_channels = {"vil", "vis", "ir069", "ir107", "lght"}
    invalid = set(channels) - valid_channels
    if invalid:
        if "glm" in invalid:
            raise ValueError(
                "'GLM' names the lightning instrument; the SEVIR channel code is "
                "'lght'. Retry with channels containing 'lght'."
            )
        raise ValueError(f"Invalid channels: {invalid}. Valid channels: {valid_channels}")

    result = {
        "status": "ok",
        "task": "load_sevir_multimodal_event",
        "event_id": event_id,
        "channels_requested": channels,
        "channels_loaded": [],
        "channels_unavailable": [],
        "data": {},
    }

    for channel in channels:
        try:
            record = _catalog_event_by_type(event_id, channel, file_name, file_index)

            # Load the array
            frames, source = _load_array_by_type(record)

            # Handle frame selection
            start = start_frame
            end = end_frame
            minute_offsets = record.get("minute_offsets", [])

            if minutes_before is not None or minutes_after is not None:
                if not minute_offsets:
                    raise ValueError(f"Channel {channel}: minute_offsets not available for time-based selection")
                offsets_array = np.array(minute_offsets)
                if minutes_before is not None:
                    start = int(np.searchsorted(offsets_array, -minutes_before, side="left"))
                if minutes_after is not None:
                    end = int(np.searchsorted(offsets_array, minutes_after, side="right"))

            start = max(0, min(start, len(frames)))
            end = max(start, min(end, len(frames)))
            selected = frames[start:end]
            offsets = minute_offsets[start:end] if minute_offsets else []

            # Build frame times
            from datetime import datetime, timedelta
            anchor_time_str = record.get("time_utc", "")
            if anchor_time_str:
                anchor = datetime.fromisoformat(anchor_time_str.replace("Z", "+00:00"))
                times = [
                    (anchor + timedelta(minutes=int(offset))).isoformat() + "Z"
                    for offset in offsets
                ]
            else:
                times = []

            # Store channel data
            channel_data = {
                "frames": np.ascontiguousarray(selected, dtype=np.float32),
                "shape": list(selected.shape),
                "source": source,
                "file_index": record.get("file_index"),
                "frame_times_utc": times,
                "relative_minutes": list(offsets),
                "frame_interval": {"start_inclusive": start, "end_exclusive": end},
                "metadata": CHANNEL_METADATA.get(channel, {}),
                "catalog_record": _full_catalog_record(record),
            }

            result["data"][channel] = channel_data
            result["channels_loaded"].append(channel)

        except Exception as error:
            result["channels_unavailable"].append({
                "channel": channel,
                "reason": str(error),
            })

    if result["channels_unavailable"] and not result["channels_loaded"]:
        result["status"] = "error"
    elif result["channels_unavailable"]:
        result["status"] = "partial"

    result["interpretation_limit"] = (
        "Multi-modal data loaded from SEVIR. Each channel provides complementary information: "
        "VIL shows radar echo intensity, VIS shows cloud structure (daytime only), "
        "IR069 shows mid-level water vapor, IR107 shows cloud-top temperature, "
        "LGHT shows lightning flash density. Cross-channel analysis can reveal convective development."
    )

    return result


def get_available_channels_for_event(event_id: str) -> dict[str, Any]:
    """Query which channels are available for a given SEVIR event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.

    Returns
    -------
    dict
        Dictionary listing available and unavailable channels with metadata.
    """
    all_channels = ["vil", "vis", "ir069", "ir107", "lght"]
    available = []
    unavailable = []

    for channel in all_channels:
        try:
            record = _catalog_event_by_type(event_id, channel, "", None)
            available.append({
                "channel": channel,
                "file_name": record.get("file_name"),
                "time_utc": record.get("time_utc"),
                "metadata": CHANNEL_METADATA.get(channel, {}),
            })
        except Exception as error:
            unavailable.append({
                "channel": channel,
                "reason": str(error),
            })

    return {
        "status": "ok",
        "task": "get_available_channels",
        "event_id": event_id,
        "available_channels": available,
        "unavailable_channels": unavailable,
        "total_available": len(available),
    }
