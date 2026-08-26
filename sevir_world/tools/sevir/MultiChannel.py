"""
Multi-channel SEVIR data loading for IR069, IR107, VIS, and LGHT.

Provides unified loading functions for satellite and lightning channels
to support multi-modal weather event analysis alongside VIL radar data.
"""

from typing import Any
import numpy as np
from sevir_world.tools.sevir.Catalog import (
    _catalog_vil_index,
    _catalog_subject_key,
    _load_array_by_type,
    _frame_times,
    ROOT,
)


SUPPORTED_CHANNELS = {"ir069", "ir107", "vis", "lght"}


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
        raise ValueError(f"Unsupported channel '{channel}'. Must be one of {SUPPORTED_CHANNELS}.")

    # Use the new unified catalog loader that supports all image types
    from sevir_world.tools.sevir.Catalog import _catalog_event_by_type

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
            "frame_times_utc": _frame_times(record["time_utc"], record["minute_offsets"]),
            "reason": str(error),
        }

    # Apply time or frame windowing
    if minutes_before is not None or minutes_after is not None:
        before = max(0, int(minutes_before or 0))
        after = max(0, int(minutes_after or 0))
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

    # Compute channel-specific statistics
    frame_statistics = []
    for position, frame in enumerate(selected, start=start):
        if channel == "lght":
            # Lightning: count events per frame
            stats = {
                "frame_index": position,
                "event_count": int(np.sum(frame > 0)),
                "max_intensity": float(np.max(frame)),
                "total_energy": float(np.sum(frame)),
            }
        elif channel in ("ir069", "ir107"):
            # IR channels: temperature statistics (encoded as int16)
            valid = frame[frame != -9999]  # typical IR missing value
            if len(valid) > 0:
                stats = {
                    "frame_index": position,
                    "min_temp": int(np.min(valid)),
                    "max_temp": int(np.max(valid)),
                    "mean_temp": float(np.mean(valid)),
                    "valid_pixels": len(valid),
                }
            else:
                stats = {
                    "frame_index": position,
                    "min_temp": None,
                    "max_temp": None,
                    "mean_temp": None,
                    "valid_pixels": 0,
                }
        elif channel == "vis":
            # Visible channel: reflectance statistics
            valid = frame[frame != -9999]
            if len(valid) > 0:
                stats = {
                    "frame_index": position,
                    "min_reflectance": int(np.min(valid)),
                    "max_reflectance": int(np.max(valid)),
                    "mean_reflectance": float(np.mean(valid)),
                    "valid_pixels": len(valid),
                }
            else:
                stats = {
                    "frame_index": position,
                    "min_reflectance": None,
                    "max_reflectance": None,
                    "mean_reflectance": None,
                    "valid_pixels": 0,
                }
        else:
            stats = {"frame_index": position}

        frame_statistics.append(stats)

    return {
        "status": "success",
        "task": "load_sevir_channel_event",
        "event_id": record["id"],
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
        "frame_statistics": frame_statistics,
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
            from sevir_world.tools.sevir.Catalog import load_sevir_vil_event
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

        if result.get("status") == "success":
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
