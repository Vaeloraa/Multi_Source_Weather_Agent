"""
Multi-channel SEVIR data loading for IR069, IR107, VIS, and LGHT.

Provides unified loading functions for satellite and lightning channels
to support multi-modal weather event analysis alongside VIL radar data.
"""

from typing import Any
from weather_agent.errors import error_record, is_transient_transport_error
import numpy as np
from weather_agent.tools.sevir.repository import (
    _catalog_vil_index,
    _catalog_subject_key,
    _load_array_by_type,
    _frame_times,
    ROOT,
)


from .channels import SUPPORTED_CHANNELS


def _get_channel_catalog_record(
    event_id: str,
    channel: str,
    file_name: str = "",
    file_index: int | None = None,
) -> dict[str, Any]:
    """Find catalog record for a specific channel of an event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    channel : str
        Channel type: 'ir069', 'ir107', 'vis', or 'lght'.
    file_name : str, optional
        Optional source-file discriminator.
    file_index : int, optional
        Optional dataset-index discriminator.

    Returns
    -------
    dict
        Catalog record for the requested channel.
    """
    if channel not in SUPPORTED_CHANNELS:
        if str(channel).strip().lower() == "glm":
            raise ValueError(
                "'GLM' names the lightning instrument; the SEVIR channel code is "
                "'lght'. Retry with channel='lght'."
            )
        raise ValueError(f"Unsupported channel '{channel}'. Must be one of {SUPPORTED_CHANNELS}.")

    # Use the new unified catalog loader that supports all image types
    from weather_agent.tools.sevir.repository import _catalog_event_by_type

    return _catalog_event_by_type(event_id, channel, file_name, file_index)


def load_sevir_channel_event(
    event_id: str,
    channel: str,
    start_frame: int = 0,
    end_frame: int = 49,
    file_name: str = "",
    file_index: int | None = None,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
) -> dict[str, Any]:
    """Load a bounded frame interval for one SEVIR channel event.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    channel : str
        Channel type: 'ir069', 'ir107', 'vis', or 'lght'.
    start_frame : int, default 0
        Inclusive zero-based first frame.
    end_frame : int, default 49
        Inclusive zero-based last frame.
    file_name : str, optional
        Optional source-file discriminator.
    file_index : int, optional
        Optional dataset-index discriminator.
    minutes_before : int, optional
        Load frames within this many minutes before reference time.
    minutes_after : int, optional
        Load frames within this many minutes after reference time.

    Returns
    -------
    dict
        Frame array, relative minutes, dimensions and event provenance.

    Examples
    --------
    Load IR107 cloud-top temperature for an event::

        cube = load_sevir_channel_event(event_id, "ir107", 0, 12)

    Load LGHT lightning data::

        lght = load_sevir_channel_event(event_id, "lght", minutes_before=60, minutes_after=0)
    """
    record = _get_channel_catalog_record(event_id, channel, file_name=file_name, file_index=file_index)
    from .repository import spatial_metadata_for_record

    if any(value is not None and int(value) < 0 for value in (minutes_before, minutes_after)):
        raise ValueError("minutes_before and minutes_after must be non-negative")
    try:
        frames, source = _load_array_by_type(record)
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "status": "data_unavailable",
            "task": "load_sevir_channel_event",
            "event_id": record["id"],
            "channel": channel,
            "pixel_data_loaded": False,
            "required_hdf5": record["file_name"],
            "file_index": record["file_index"],
            "metadata_available": True,
            "spatial_metadata": spatial_metadata_for_record(record),
            "frame_times_utc": _frame_times(record["time_utc"], record["minute_offsets"]),
            "reason": str(error),
            "error": error_record(error),
            "retryable": is_transient_transport_error(error),
        }

    # Apply time or frame windowing
    if minutes_before is not None or minutes_after is not None:
        before = int(minutes_before or 0)
        after = int(minutes_after or 0)
        positions = [
            index for index, offset in enumerate(record["minute_offsets"])
            if -before <= int(offset) <= after
        ]
        if not positions:
            raise ValueError(
                f"No observations fall in the requested {-before}..{after} minute window."
            )
        start, end = positions[0], positions[-1] + 1
    else:
        start = max(0, int(start_frame))
        end = min(int(end_frame) + 1, int(frames.shape[0]))

    if start >= end:
        raise ValueError(f"Invalid frame interval [{start_frame}, {end_frame}].")

    selected = frames[start:end]
    offsets = record["minute_offsets"][start:end]
    times = _frame_times(record["time_utc"], offsets)

    return {
        "status": "success",
        "task": "load_sevir_channel_event",
        "event_id": record["id"],
        "event_anchor_time_utc": record["time_utc"],
        "channel": channel,
        "file_index": record["file_index"],
        "subject_key": _catalog_subject_key(record),
        "shape": list(selected.shape),
        "dtype": str(selected.dtype),
        "num_frames_loaded": len(selected),
        "start_frame": start,
        "end_frame": end - 1,
        "minute_offsets": [int(x) for x in offsets],
        "frame_times_utc": times,
        # Keep pixels in-process for the request-bound public capability. The
        # worker transport still summarizes xarray values instead of embedding
        # these cubes in the model response.
        "frames": selected,
        "values": selected,
        "relative_minutes": [int(x) for x in offsets],
        "spatial_metadata": spatial_metadata_for_record(record),
        "missing_value": -9999,
        "data_source": source,
        "pixel_data_loaded": True,
        "pixel_data_returned_to_llm": False,
        "pixel_data_available_to_generated_code": True,
        "interpretation_limit": (
            f"The HDF5 {channel} pixels were read for computation, but the full array is not embedded "
            "in the tool response. Local HDF5, remote HDF5, or an event NPZ cache may be used."
        ),
    }


def load_sevir_multimodal_event(
    event_id: str,
    channels: list[str] | None = None,
    start_frame: int = 0,
    end_frame: int = 49,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
) -> dict[str, Any]:
    """Load multiple channels for one SEVIR event simultaneously.

    Parameters
    ----------
    event_id : str
        SEVIR event identifier.
    channels : list of str, optional
        Channels to load. Defaults to ['vil', 'ir107', 'lght'].
    start_frame, end_frame : int
        Frame range to load.
    minutes_before, minutes_after : int, optional
        Time window relative to reference time.

    Returns
    -------
    dict
        Dictionary with one entry per channel, plus metadata.

    Examples
    --------
    Load VIL + IR107 + Lightning for comprehensive analysis::

        data = load_sevir_multimodal_event(
            event_id,
            channels=["vil", "ir107", "lght"],
            minutes_before=60,
            minutes_after=0
        )
    """
    if channels is None:
        channels = ["vil", "ir107", "lght"]

    results = {}
    available_channels = []
    unavailable_channels = []

    for channel in channels:
        if channel == "vil":
            from weather_agent.tools.sevir.repository import load_sevir_vil_event
            result = load_sevir_vil_event(
                event_id=event_id,
                start_frame=start_frame,
                end_frame=end_frame,
                minutes_before=minutes_before,
                minutes_after=minutes_after,
            )
        else:
            result = load_sevir_channel_event(
                event_id=event_id,
                channel=channel,
                start_frame=start_frame,
                end_frame=end_frame,
                minutes_before=minutes_before,
                minutes_after=minutes_after,
            )

        results[channel] = result

        # The established VIL loader uses ``ok`` while the newer satellite
        # loaders use ``success``. Both represent a completed pixel read.
        if result.get("status") in {"ok", "success"}:
            available_channels.append(channel)
        else:
            unavailable_channels.append(channel)

    return {
        "status": "success" if len(available_channels) == len(channels) else "partial",
        "task": "load_sevir_multimodal_event",
        "event_id": event_id,
        "channels_requested": channels,
        "channels_available": available_channels,
        "channels_unavailable": unavailable_channels,
        "channel_data": results,
    }
