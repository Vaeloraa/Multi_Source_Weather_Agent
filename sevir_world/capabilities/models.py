from __future__ import annotations

"""Small Python values returned by the four public capabilities.

These classes deliberately contain no transport status, schema envelope, or
legacy adapter payload.  A failed operation raises an exception; a successful
operation returns a value whose attributes can be used directly by NumPy code.
"""

from dataclasses import dataclass, fields
from typing import Any, Iterator

import numpy as np


class CapabilityValue:
    """Base with concise representation and predictable field iteration."""

    def items(self) -> Iterator[tuple[str, Any]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def __repr__(self) -> str:
        parts: list[str] = []
        for name, value in self.items():
            if isinstance(value, np.ndarray):
                shown = f"ndarray(shape={value.shape}, dtype={value.dtype})"
            elif isinstance(value, (list, tuple)) and len(value) > 8:
                shown = f"{type(value).__name__}(len={len(value)})"
            elif isinstance(value, dict) and len(value) > 8:
                shown = f"dict(keys={list(value)[:8]}...)"
            else:
                shown = repr(value)
            parts.append(f"{name}={shown}")
        return f"{type(self).__name__}({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Bounds(CapabilityValue):
    south: float
    west: float
    north: float
    east: float


@dataclass(frozen=True, slots=True)
class Event(CapabilityValue):
    id: str
    time_utc: str
    event_type: str
    frame_times_utc: tuple[str, ...]
    minute_offsets: np.ndarray
    bounds: Bounds | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CatalogEvent(CapabilityValue):
    id: str
    time_utc: str
    event_type: str
    data_min: float | None
    data_max: float | None
    missing_fraction: float | None
    bounds: Bounds | None
    distance_km: float | None
    local_data: bool
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationCube(CapabilityValue):
    event_id: str
    values: np.ndarray
    minutes: np.ndarray
    missing_value: int
    dimensions: tuple[str, str, str] = ("time", "row", "column")


@dataclass(frozen=True, slots=True)
class UploadMatch(CapabilityValue):
    upload_id: str
    event_id: str | None
    confidence: float | None
    method: str


@dataclass(frozen=True, slots=True)
class PixelLocation(CapabilityValue):
    row: int
    column: int
    latitude: float
    longitude: float
    projected_x: float | None = None
    projected_y: float | None = None


@dataclass(frozen=True, slots=True)
class Location(CapabilityValue):
    event_id: str
    center: tuple[float, float]
    bounds: Bounds
    image_shape: tuple[int, int]
    projection: str


@dataclass(frozen=True, slots=True)
class Place(CapabilityValue):
    query: str
    latitude: float
    longitude: float
    name: str
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AdministrativeArea(CapabilityValue):
    latitude: float
    longitude: float
    country: str | None
    state: str | None
    state_code: str | None
    county: str | None
    county_fips: str | None


@dataclass(frozen=True, slots=True)
class RegionCoverage(CapabilityValue):
    event_id: str
    pixel_count: int
    pixel_box: tuple[int, int, int, int]
    bounds: Bounds
    center: PixelLocation
    areas: tuple[dict[str, Any], ...]
    unresolved_pixels: int


@dataclass(frozen=True, slots=True)
class Forecast(CapabilityValue):
    subject_id: str
    model: str
    mean: np.ndarray
    members: np.ndarray
    lead_minutes: np.ndarray
    ground_truth: np.ndarray | None
    member_seeds: tuple[int, ...]
    cache_hit: bool

    @property
    def horizon_minutes(self) -> int:
        return int(self.lead_minutes[-1]) if self.lead_minutes.size else 0


@dataclass(frozen=True, slots=True)
class EnvironmentFields(CapabilityValue):
    event_id: str | None
    analysis_time_utc: str
    latitude: np.ndarray
    longitude: np.ndarray
    era5: dict[str, Any]
    forecast: dict[int, dict[str, Any]]
    lead_hours: np.ndarray
    cache_hit: bool

