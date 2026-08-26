"""Test multi-modal SEVIR channel loading (VIS, IR069, IR107, LGHT)."""

from __future__ import annotations

import pytest
import numpy as np

from sevir_world.tools.sevir.MultiChannel import (
    load_sevir_channel_event,
    load_sevir_multimodal_event,
    SUPPORTED_CHANNELS,
)


def test_supported_channels_defined():
    """Verify that all four new channels are recognized."""
    assert "ir069" in SUPPORTED_CHANNELS
    assert "ir107" in SUPPORTED_CHANNELS
    assert "vis" in SUPPORTED_CHANNELS
    assert "lght" in SUPPORTED_CHANNELS


def test_load_ir107_channel():
    """Load IR107 (cloud-top temperature) for a known event."""
    event_id = "R19062708588472"

    result = load_sevir_channel_event(
        event_id=event_id,
        channel="ir107",
        start_frame=0,
        end_frame=12,
    )

    # Check basic structure
    assert result["task"] == "load_sevir_channel_event"
    assert result["event_id"] == event_id
    assert result["channel"] == "ir107"

    # Verify we got frame statistics
    assert "frame_statistics" in result
    assert len(result["frame_statistics"]) > 0

    # IR107 stats should include temperature fields
    first_stat = result["frame_statistics"][0]
    if result.get("status") == "success":
        assert "min_temp" in first_stat or "valid_pixels" in first_stat


def test_load_lght_channel():
    """Load LGHT (lightning flash density) for a known event."""
    event_id = "R19062708588472"

    result = load_sevir_channel_event(
        event_id=event_id,
        channel="lght",
        minutes_before=60,
        minutes_after=0,
    )

    assert result["task"] == "load_sevir_channel_event"
    assert result["channel"] == "lght"

    # Lightning stats should include event counts
    if result.get("status") == "success":
        first_stat = result["frame_statistics"][0]
        assert "event_count" in first_stat
        assert "max_intensity" in first_stat


def test_load_multimodal_vil_ir_lght():
    """Load VIL + IR107 + LGHT simultaneously for comprehensive analysis."""
    event_id = "R19062708588472"

    result = load_sevir_multimodal_event(
        event_id=event_id,
        channels=["vil", "ir107", "lght"],
        minutes_before=60,
        minutes_after=0,
    )

    assert result["task"] == "load_sevir_multimodal_event"
    assert result["event_id"] == event_id
    assert result["channels_requested"] == ["vil", "ir107", "lght"]

    # Check that we got data for at least one channel
    assert "channel_data" in result
    assert len(result["channel_data"]) > 0

    # At least VIL should be available
    assert "vil" in result["channel_data"]


def test_multimodal_default_channels():
    """Verify default channel selection is VIL + IR107 + LGHT."""
    event_id = "R19062708588472"

    result = load_sevir_multimodal_event(
        event_id=event_id,
        minutes_before=30,
        minutes_after=0,
    )

    # Default should request vil, ir107, lght
    assert "vil" in result["channels_requested"]
    assert "ir107" in result["channels_requested"]
    assert "lght" in result["channels_requested"]


def test_channel_unavailable_graceful():
    """Verify graceful handling when a channel is not available."""
    event_id = "R19062708588472"

    # Request a channel that might not exist for this event
    result = load_sevir_channel_event(
        event_id=event_id,
        channel="vis",  # VIS might be unavailable at night
        minutes_before=60,
        minutes_after=0,
    )

    # Should return a structured response even if unavailable
    assert "task" in result
    assert "channel" in result
    assert result["channel"] == "vis"

    # Check status field
    assert "status" in result
    if result["status"] == "data_unavailable":
        assert "reason" in result
        assert "metadata_available" in result


def test_multimodal_partial_availability():
    """Verify multimodal loading when some channels are unavailable."""
    event_id = "R19062708588472"

    # Request all channels including potentially unavailable ones
    result = load_sevir_multimodal_event(
        event_id=event_id,
        channels=["vil", "vis", "ir069", "ir107", "lght"],
        minutes_before=30,
        minutes_after=0,
    )

    # Should distinguish between available and unavailable
    assert "channels_available" in result
    assert "channels_unavailable" in result

    # Status should be "partial" if some channels failed
    if len(result["channels_unavailable"]) > 0:
        assert result["status"] in ("partial", "success")


def test_ir069_water_vapor_channel():
    """Load IR069 (water vapor) channel."""
    event_id = "R19062708588472"

    result = load_sevir_channel_event(
        event_id=event_id,
        channel="ir069",
        start_frame=0,
        end_frame=12,
    )

    assert result["channel"] == "ir069"

    # IR069 should have temperature statistics like IR107
    if result.get("status") == "success":
        first_stat = result["frame_statistics"][0]
        assert "min_temp" in first_stat or "valid_pixels" in first_stat


def test_invalid_channel_rejected():
    """Verify that invalid channel names are rejected."""
    event_id = "R19062708588472"

    with pytest.raises(ValueError, match="Unsupported channel"):
        load_sevir_channel_event(
            event_id=event_id,
            channel="invalid_channel",
            start_frame=0,
            end_frame=12,
        )


@pytest.mark.parametrize("channel", ["ir069", "ir107", "vis", "lght"])
def test_each_channel_structure(channel):
    """Verify each channel returns consistent structure."""
    event_id = "R19062708588472"

    result = load_sevir_channel_event(
        event_id=event_id,
        channel=channel,
        start_frame=0,
        end_frame=5,
    )

    # All channels should return these fields
    assert "task" in result
    assert "event_id" in result
    assert "channel" in result
    assert "status" in result

    # If successful, should have frame statistics
    if result.get("status") == "success":
        assert "frame_statistics" in result
        assert "shape" in result
        assert "minute_offsets" in result
        assert "frame_times_utc" in result
