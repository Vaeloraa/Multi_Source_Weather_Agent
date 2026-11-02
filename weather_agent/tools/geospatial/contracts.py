from __future__ import annotations

"""Canonical spatial values shared by requests, data references and tools."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Validated WGS84 bounds in south-west-north-east order."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (-90.0 <= float(self.south) < float(self.north) <= 90.0):
            raise ValueError("bbox must satisfy -90 <= south < north <= 90")
        if not (-180.0 <= float(self.west) < float(self.east) <= 180.0):
            raise ValueError("bbox must satisfy -180 <= west < east <= 180")

    @classmethod
    def from_value(cls, value: Any) -> "BoundingBox | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("bbox must contain south, west, north and east")
        required = {"south", "west", "north", "east"}
        if set(value) != required:
            raise ValueError("bbox must contain exactly south, west, north and east")
        return cls(**{name: float(value[name]) for name in required})

    def as_dict(self) -> dict[str, float]:
        return {
            "south": float(self.south), "west": float(self.west),
            "north": float(self.north), "east": float(self.east),
        }


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """One WGS84 point, optionally linked to a grid cell."""

    latitude: float
    longitude: float
    row: int | None = None
    column: int | None = None
    projected_x: float | None = None
    projected_y: float | None = None
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if not -90 <= float(self.latitude) <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= float(self.longitude) <= 180:
            raise ValueError("longitude must be between -180 and 180")


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One zero-based grid cell and its WGS84 centre."""

    row: int
    column: int
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class Polygon:
    """WGS84 polygon expressed as longitude/latitude vertices."""

    coordinates: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.coordinates) < 3:
            raise ValueError("polygon requires at least three vertices")
        for longitude, latitude in self.coordinates:
            if not -180 <= float(longitude) <= 180 or not -90 <= float(latitude) <= 90:
                raise ValueError("polygon coordinate is outside WGS84")

    @classmethod
    def from_value(cls, value: Any) -> "Polygon":
        raw: Sequence[Any]
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = value.get("coordinates", ())
            if raw and isinstance(raw[0], Sequence) and raw[0] and isinstance(raw[0][0], Sequence):
                raw = raw[0]
        else:
            raw = value
        return cls(tuple((float(item[0]), float(item[1])) for item in raw))


@dataclass(frozen=True, slots=True)
class Region:
    """A source-independent named, bounded or polygonal WGS84 region."""

    bbox: BoundingBox
    polygon: Polygon | None = None
    name: str = ""
    level: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)


def region_from_value(value: Any) -> Region:
    """Normalize bbox, polygon, GeoJSON or Region input."""

    if isinstance(value, Region):
        return value
    if isinstance(value, BoundingBox):
        return Region(value)
    if not isinstance(value, Mapping):
        polygon = Polygon.from_value(value)
        xs = [item[0] for item in polygon.coordinates]
        ys = [item[1] for item in polygon.coordinates]
        return Region(BoundingBox(min(ys), min(xs), max(ys), max(xs)), polygon)
    if {"south", "west", "north", "east"}.issubset(value):
        bbox = BoundingBox.from_value({name: value[name] for name in ("south", "west", "north", "east")})
        assert bbox is not None
        return Region(bbox, name=str(value.get("name") or ""), level=str(value.get("level") or ""))
    geometry = value.get("geometry", value)
    polygon = Polygon.from_value(geometry)
    xs = [item[0] for item in polygon.coordinates]
    ys = [item[1] for item in polygon.coordinates]
    return Region(
        BoundingBox(min(ys), min(xs), max(ys), max(xs)), polygon,
        name=str(value.get("name") or ""), level=str(value.get("level") or ""),
        properties=dict(value.get("properties") or {}),
    )


__all__ = ["BoundingBox", "GeoPoint", "GridPoint", "Polygon", "Region", "region_from_value"]
