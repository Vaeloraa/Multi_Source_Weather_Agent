from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..backends import geospatial

from .base import WorldComponent, resolve_event_id


BOUNDARY_ROOT = Path(__file__).resolve().parents[2] / "data" / "boundaries"


def _point_in_ring(longitude: float, latitude: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > latitude) != (y2 > latitude):
            crossing = (x2 - x1) * (latitude - y1) / ((y2 - y1) or 1e-12) + x1
            if longitude < crossing:
                inside = not inside
        previous = current
    return inside


@lru_cache(maxsize=4)
def _load_kmz(path_text: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_text)
    if not path.exists():
        return ()
    with zipfile.ZipFile(path) as archive:
        kml_name = next(name for name in archive.namelist() if name.lower().endswith(".kml"))
        root = ElementTree.fromstring(archive.read(kml_name))
    namespace = {"k": "http://www.opengis.net/kml/2.2"}
    features: list[dict[str, Any]] = []
    for placemark in root.findall(".//k:Placemark", namespace):
        fields = {
            item.attrib.get("name", ""): (item.text or "").strip()
            for item in placemark.findall(".//k:SimpleData", namespace)
        }
        polygons: list[list[tuple[float, float]]] = []
        for coordinates in placemark.findall(
            ".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", namespace
        ):
            ring = []
            for token in (coordinates.text or "").split():
                parts = token.split(",")
                if len(parts) >= 2:
                    ring.append((float(parts[0]), float(parts[1])))
            if len(ring) >= 3:
                polygons.append(ring)
        if not polygons:
            continue
        xs = [point[0] for polygon in polygons for point in polygon]
        ys = [point[1] for polygon in polygons for point in polygon]
        features.append(
            {
                "fields": fields,
                "polygons": polygons,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            }
        )
    return tuple(features)


def _find_feature(path: Path, latitude: float, longitude: float) -> dict[str, Any] | None:
    for feature in _load_kmz(str(path)):
        min_lon, min_lat, max_lon, max_lat = feature["bbox"]
        if not (min_lon <= longitude <= max_lon and min_lat <= latitude <= max_lat):
            continue
        if any(_point_in_ring(longitude, latitude, ring) for ring in feature["polygons"]):
            return feature["fields"]
    return None


class SevirGeolocator(WorldComponent):
    """Convert SEVIR grid coordinates and optionally resolve US state/county.

    Pixel conversion uses the projection and event geometry stored in CATALOG.
    Administrative lookup uses bundled US Census cartographic boundaries and
    describes the polygon containing the point; it does not identify a city.
    """

    def locate(
        self, event_id: Any, row: int | None = None, column: int | None = None
    ) -> dict[str, Any]:
        """Return event bounds, center, and optional projected pixel location."""
        event_id = resolve_event_id(event_id)
        arguments = {"event_id": event_id, "row": row, "column": column}
        return self._call("geolocator.locate", arguments, geospatial.geolocate_sevir_event)

    def pixel_to_latlon(self, event_id: Any, row: int, column: int) -> dict[str, Any]:
        """Convert a zero-based raster row/column to projected WGS84 lat/lon."""
        location = self.locate(event_id, row=row, column=column)
        return location["pixel_location"]

    def reverse_geocode(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Resolve a WGS84 point to offline Census state and county polygons."""
        latitude = float(latitude)
        longitude = float(longitude)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude/longitude is outside the WGS84 range")
        state_path = BOUNDARY_ROOT / "cb_2025_us_state_20m.kmz"
        county_path = BOUNDARY_ROOT / "cb_2025_us_county_20m.kmz"
        if not state_path.exists() or not county_path.exists():
            result = {
                "status": "boundary_data_unavailable",
                "latitude": latitude,
                "longitude": longitude,
                "required_files": [state_path.name, county_path.name],
                "boundary_root": str(BOUNDARY_ROOT),
            }
        else:
            state = _find_feature(state_path, latitude, longitude)
            county = _find_feature(county_path, latitude, longitude)
            result = {
                "status": "ok" if state else "outside_us_boundaries",
                "latitude": latitude,
                "longitude": longitude,
                "country": "United States" if state else None,
                "state": state.get("NAME") if state else None,
                "state_code": state.get("STUSPS") if state else None,
                "state_fips": state.get("STATEFP") if state else None,
                "county": county.get("NAME") if county else None,
                "county_fips": county.get("GEOID") if county else None,
                "source": "US Census 2025 cartographic boundary files, 1:20m",
                "interpretation_limit": "Polygon lookup only; not a city or street geocoder.",
            }
        self._recorder(
            "geolocator.reverse_geocode",
            {"latitude": latitude, "longitude": longitude},
            result,
        )
        return result

    def locate_admin(self, event_id: Any, row: int = 192, column: int = 192) -> dict[str, Any]:
        """Resolve an event pixel to WGS84 and offline state/county."""
        event_id = resolve_event_id(event_id)
        pixel = self.pixel_to_latlon(event_id, row, column)
        admin = self.reverse_geocode(pixel["latitude"], pixel["longitude"])
        return {"event_id": event_id, "pixel_location": pixel, "administrative_area": admin}
