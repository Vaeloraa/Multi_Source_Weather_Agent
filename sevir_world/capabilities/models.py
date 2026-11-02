from __future__ import annotations

"""Small semantic metadata values used at capability boundaries.

Scientific grids and tables are returned as xarray or pandas objects. These
records are deliberately flat, immutable metadata only; they do not emulate
NumPy or xarray APIs.
"""

from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd


class CapabilityValue:
    """Provide predictable field iteration for small metadata records."""

    def items(self) -> Iterator[tuple[str, Any]]:
        for item in fields(self):
            yield item.name, getattr(self, item.name)


@dataclass(frozen=True, slots=True)
class Event(CapabilityValue):
    """Metadata and timing axes for one SEVIR event."""

    subject_key: str
    id: str
    event_anchor_time_utc: str
    event_type: str
    frame_times_utc: tuple[str, ...]
    minute_offsets: np.ndarray
    south: float | None
    west: float | None
    north: float | None
    east: float | None
    spatial_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class UploadMatch(CapabilityValue):
    """Result of matching an uploaded sequence to a catalog event."""

    upload_id: str
    subject_key: str | None
    event_id: str | None
    confidence: float | None
    method: str


@dataclass(frozen=True, slots=True)
class Place(CapabilityValue):
    """Resolved place with a standard pandas candidate table."""

    query: str
    status: str
    name: str
    latitude: float
    longitude: float
    candidates: pd.DataFrame


@dataclass(frozen=True, slots=True)
class AdministrativeArea(CapabilityValue):
    """Administrative names resolved for one WGS84 point."""

    latitude: float
    longitude: float
    status: str
    match_method: str
    country: str | None
    state: str | None
    state_code: str | None
    county: str | None
    county_fips: str | None
    continent: str | None = None
    water_body: str | None = None

