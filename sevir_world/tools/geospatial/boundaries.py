from __future__ import annotations

"""Boundary lookup independent of weather datasets and grid formats."""

from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
import zipfile

from .contracts import BoundingBox, Polygon, Region


BOUNDARY_ROOT = Path(__file__).resolve().parents[3] / "data" / "boundaries"


def _inside(longitude: float, latitude: float, ring: tuple[tuple[float, float], ...]) -> bool:
    result = False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if (y1 > latitude) != (y2 > latitude):
            result ^= longitude < (x2 - x1) * (latitude - y1) / ((y2 - y1) or 1e-12) + x1
        x1, y1 = x2, y2
    return result


@lru_cache(maxsize=8)
def _features(level: str) -> tuple[dict[str, Any], ...]:
    files = {
        "state": "cb_2025_us_state_20m.kmz",
        "county": "cb_2025_us_county_20m.kmz",
    }
    if level not in files:
        return ()
    path = BOUNDARY_ROOT / files[level]
    if not path.is_file():
        return ()
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.lower().endswith(".kml"))
        root = ElementTree.fromstring(archive.read(name))
    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    result: list[dict[str, Any]] = []
    for placemark in root.findall(".//k:Placemark", namespace):
        fields = {
            item.attrib.get("name", ""): (item.text or "").strip()
            for item in placemark.findall(".//k:SimpleData", namespace)
        }
        polygons = []
        for node in placemark.findall(
            ".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", namespace
        ):
            ring = tuple(
                (float(parts[0]), float(parts[1]))
                for token in (node.text or "").split()
                if len(parts := token.split(",")) >= 2
            )
            if len(ring) >= 3:
                polygons.append(ring)
        if polygons:
            xs = [point[0] for polygon in polygons for point in polygon]
            ys = [point[1] for polygon in polygons for point in polygon]
            result.append({
                "fields": fields, "polygons": tuple(polygons),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            })
    return tuple(result)


def point_to_location(latitude: float, longitude: float) -> dict[str, Any]:
    """Resolve a WGS84 point to available geographic feature levels."""

    latitude, longitude = float(latitude), float(longitude)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside WGS84")
    matches: dict[str, dict[str, str] | None] = {}
    for level in ("state", "county"):
        match = None
        for feature in _features(level):
            west, south, east, north = feature["bbox"]
            if west <= longitude <= east and south <= latitude <= north and any(
                _inside(longitude, latitude, polygon) for polygon in feature["polygons"]
            ):
                match = feature["fields"]
                break
        matches[level] = match
    state, county = matches["state"], matches["county"]
    return {
        "status": "ok" if state else "outside_available_boundaries",
        "latitude": latitude, "longitude": longitude,
        "country": "United States" if state else None,
        "state": state.get("NAME") if state else None,
        "state_code": state.get("STUSPS") if state else None,
        "county": county.get("NAME") if county else None,
        "county_fips": county.get("GEOID") if county else None,
        "continent": "North America" if state else None,
        "water_body": None,
        "provenance": {"source": "US Census 2025 cartographic boundaries, 1:20m"},
    }


def boundary(name: str, level: str | None = None) -> Region:
    """Return one named administrative boundary and its hierarchy fields."""

    query = str(name or "").strip().casefold()
    if not query:
        raise ValueError("boundary name must not be empty")
    levels = (str(level).lower(),) if level else ("county", "state")
    matches = []
    for candidate_level in levels:
        for feature in _features(candidate_level):
            fields = feature["fields"]
            names = {str(fields.get(key) or "").casefold() for key in ("NAME", "NAMELSAD", "STUSPS", "GEOID")}
            if query in names:
                matches.append((candidate_level, feature))
    if not matches:
        raise ValueError(f"boundary {name!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"boundary {name!r} is ambiguous; specify level")
    candidate_level, feature = matches[0]
    polygon = max(feature["polygons"], key=len)
    west, south, east, north = feature["bbox"]
    fields = dict(feature["fields"])
    return Region(
        BoundingBox(south, west, north, east), Polygon(polygon),
        name=str(fields.get("NAMELSAD") or fields.get("NAME") or name),
        level=candidate_level, properties=fields,
    )


__all__ = ["boundary", "point_to_location"]
