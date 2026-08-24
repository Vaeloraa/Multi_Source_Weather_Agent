from __future__ import annotations

import base64
import binascii
import csv
import json
import os
import uuid
import zipfile
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..data_access.config import *
from ..data_access.catalog import _catalog_vil_event

import h5py
import numpy as np
from pyproj import CRS, Transformer


BOUNDARY_ROOT = Path(__file__).resolve().parents[2] / "data" / "boundaries"
DEFAULT_GEOCODER_URL = "https://nominatim.openstreetmap.org/search"


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


def _points_in_ring_batch(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    ring: list[tuple[float, float]],
) -> np.ndarray:
    """Vectorized equivalent of :func:`_point_in_ring` for pixel centres."""
    longitude_values = np.asarray(longitudes, dtype=np.float64).reshape(-1)
    latitude_values = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    if longitude_values.size != latitude_values.size:
        raise ValueError("longitude and latitude arrays must have the same length")
    if longitude_values.size == 0:
        return np.zeros(0, dtype=bool)
    coordinates = np.asarray(ring, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 3 or coordinates.shape[1] < 2:
        return np.zeros(longitude_values.size, dtype=bool)
    x2 = coordinates[:, 0][None, :]
    y2 = coordinates[:, 1][None, :]
    x1 = np.roll(coordinates[:, 0], 1)[None, :]
    y1 = np.roll(coordinates[:, 1], 1)[None, :]
    point_x = longitude_values[:, None]
    point_y = latitude_values[:, None]
    crosses_latitude = (y1 > point_y) != (y2 > point_y)
    denominator = y2 - y1
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    crossing_x = (x2 - x1) * (point_y - y1) / denominator + x1
    return np.count_nonzero(crosses_latitude & (point_x < crossing_x), axis=1) % 2 == 1


@lru_cache(maxsize=4)
def _load_boundary_kmz(path_text: str) -> tuple[dict[str, Any], ...]:
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


def _find_boundary_feature(
    path: Path, latitude: float, longitude: float
) -> dict[str, Any] | None:
    for feature in _load_boundary_kmz(str(path)):
        min_lon, min_lat, max_lon, max_lat = feature["bbox"]
        if not (min_lon <= longitude <= max_lon and min_lat <= latitude <= max_lat):
            continue
        if any(_point_in_ring(longitude, latitude, ring) for ring in feature["polygons"]):
            return feature["fields"]
    return None


def _nearest_boundary_feature(
    path: Path, latitude: float, longitude: float
) -> dict[str, Any] | None:
    """Return a nearby bbox feature when point-in-polygon misses a complex ring.

    Census cartographic KMZ files can contain multipolygons and small holes.
    This fallback is deliberately marked approximate by the caller and is
    only used for a point near the overall US boundary envelope.
    """
    features = _load_boundary_kmz(str(path))
    if not features:
        return None
    min_lon = min(feature["bbox"][0] for feature in features)
    min_lat = min(feature["bbox"][1] for feature in features)
    max_lon = max(feature["bbox"][2] for feature in features)
    max_lat = max(feature["bbox"][3] for feature in features)
    if not (min_lon - 2 <= longitude <= max_lon + 2 and min_lat - 2 <= latitude <= max_lat + 2):
        return None

    def distance(feature: dict[str, Any]) -> float:
        west, south, east, north = feature["bbox"]
        dx = 0.0 if west <= longitude <= east else min(abs(longitude - west), abs(longitude - east))
        dy = 0.0 if south <= latitude <= north else min(abs(latitude - south), abs(latitude - north))
        return dx * dx + dy * dy

    return min(features, key=distance)["fields"]


def reverse_geocode_us(latitude: float, longitude: float) -> dict[str, Any]:
    """Resolve a WGS84 point against bundled US Census state/county polygons."""
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside the WGS84 range")
    state_path = BOUNDARY_ROOT / "cb_2025_us_state_20m.kmz"
    county_path = BOUNDARY_ROOT / "cb_2025_us_county_20m.kmz"
    if not state_path.exists() or not county_path.exists():
        return {
            "status": "boundary_data_unavailable",
            "capabilities": [],
            "provenance": {
                "coordinate_reference_system": "WGS84",
                "administrative_boundaries_available": False,
            },
            "latitude": latitude,
            "longitude": longitude,
            "required_files": [state_path.name, county_path.name],
        }
    state = _find_boundary_feature(state_path, latitude, longitude)
    county = _find_boundary_feature(county_path, latitude, longitude)
    state_method = "point_in_polygon"
    county_method = "point_in_polygon"
    if state is None:
        state = _nearest_boundary_feature(state_path, latitude, longitude)
        state_method = "nearby_bbox_fallback" if state else "not_found"
    if county is None and state_method == "nearby_bbox_fallback":
        county = _nearest_boundary_feature(county_path, latitude, longitude)
        county_method = "nearby_bbox_fallback" if county else "not_found"
    return {
        "status": "ok" if state and state_method == "point_in_polygon" else (
            "approximate_boundary_match" if state else "outside_us_boundaries"
        ),
        "capabilities": ["offline_state_county_polygon_lookup"],
        "latitude": latitude,
        "longitude": longitude,
        "country": "United States" if state else None,
        "state": state.get("NAME") if state else None,
        "state_code": state.get("STUSPS") if state else None,
        "state_fips": state.get("STATEFP") if state else None,
        "county": county.get("NAME") if county else None,
        "county_fips": county.get("GEOID") if county else None,
        "source": "US Census 2025 cartographic boundary files, 1:20m",
        "provenance": {
            "coordinate_reference_system": "WGS84",
            "administrative_boundary_source": "US Census 2025 cartographic boundaries, 1:20m",
            "point_in_polygon_lookup": state_method == "point_in_polygon",
            "state_lookup_method": state_method,
            "county_lookup_method": county_method,
        },
        "interpretation_limit": "Polygon lookup only; not a city or street geocoder.",
    }


def resolve_place(
    place: str,
    *,
    country_codes: str = "us",
    geocoder_url: str | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    """Resolve a user place name to WGS84 through an explicit geocoder.

    Numeric ``latitude, longitude`` input is handled locally. Text names use
    the configured geocoder (Nominatim by default) and retain the provider
    response as provenance. The caller must still decide whether the returned
    point is an adequate representation of a county, city, or other region.
    """
    text = str(place or "").strip()
    if not text:
        raise ValueError("place must not be empty")
    numeric = [token.strip() for token in text.replace(";", ",").split(",")]
    if len(numeric) == 2:
        try:
            latitude, longitude = float(numeric[0]), float(numeric[1])
            point = reverse_geocode_us(latitude, longitude)
            return {
                "status": "ok",
                "query": text,
                "latitude": latitude,
                "longitude": longitude,
                "display_name": text,
                "administrative_area": point,
                "provenance": {"source": "explicit_wgs84_coordinates"},
            }
        except ValueError:
            pass

    endpoint = str(geocoder_url or os.getenv("SEVIR_GEOCODER_URL", DEFAULT_GEOCODER_URL)).strip()
    query = urllib.parse.urlencode(
        {
            "q": text,
            "format": "jsonv2",
            "limit": "5",
            "addressdetails": "1",
            "countrycodes": country_codes,
        }
    )
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        headers={"User-Agent": "sevir-world-agent/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "geocoder_unavailable",
            "query": text,
            "reason": str(error),
            "provenance": {"source": endpoint, "network_lookup": True},
        }
    if not isinstance(payload, list) or not payload:
        return {
            "status": "not_found",
            "query": text,
            "provenance": {"source": endpoint, "network_lookup": True},
        }
    candidates = []
    for item in payload:
        try:
            candidates.append({
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "display_name": str(item.get("display_name", text)),
                "type": item.get("type"),
                "category": item.get("category"),
                "address": item.get("address", {}),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        return {
            "status": "not_found",
            "query": text,
            "provenance": {"source": endpoint, "network_lookup": True},
        }
    first = candidates[0]
    return {
        "status": "ok" if len(candidates) == 1 else "ambiguous",
        "query": text,
        **first,
        "candidates": candidates,
        "provenance": {
            "source": endpoint,
            "network_lookup": True,
            "candidate_count": len(candidates),
        },
        "interpretation_limit": "A place-name result is a representative point; use an administrative polygon for full-region masking.",
    }


@lru_cache(maxsize=128)
def _projection_context(
    projection: str,
    lower_left_longitude: float,
    lower_left_latitude: float,
    upper_right_longitude: float,
    upper_right_latitude: float,
    size_x: int,
    size_y: int,
    width_m: float,
    height_m: float,
) -> tuple[Transformer, float, float, float, float]:
    """Build projection machinery once for an event grid geometry."""
    projected_crs = CRS.from_proj4(projection)
    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_latlon = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    x_min, _ = to_projected.transform(lower_left_longitude, lower_left_latitude)
    _, y_max = to_projected.transform(upper_right_longitude, upper_right_latitude)
    return (
        to_latlon,
        float(x_min),
        float(y_max),
        float(width_m) / int(size_x),
        float(height_m) / int(size_y),
    )


def _pixel_centres_latlon(
    record: dict[str, Any],
    rows: np.ndarray | list[int],
    columns: np.ndarray | list[int],
) -> dict[str, Any]:
    """Transform many zero-based pixel centres in one pyproj call."""
    size_x = int(record["size_x"])
    size_y = int(record["size_y"])
    width_m = float(record["width_m"])
    height_m = float(record["height_m"])
    projection = str(record["proj"] or "")
    if not all((size_x, size_y, width_m, height_m, projection)):
        raise ValueError(f"SEVIR event {record['id']} has incomplete projection metadata.")
    row_values = np.asarray(rows, dtype=np.int64).reshape(-1)
    column_values = np.asarray(columns, dtype=np.int64).reshape(-1)
    if row_values.size != column_values.size:
        raise ValueError("rows and columns must have the same length")
    invalid = (
        (row_values < 0) | (row_values >= size_y)
        | (column_values < 0) | (column_values >= size_x)
    )
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"Pixel (row={int(row_values[first])}, column={int(column_values[first])}) "
            f"is outside image shape ({size_y}, {size_x})."
        )
    to_latlon, x_min, y_max, pixel_width_m, pixel_height_m = _projection_context(
        projection,
        float(record["llcrnrlon"]),
        float(record["llcrnrlat"]),
        float(record["urcrnrlon"]),
        float(record["urcrnrlat"]),
        size_x,
        size_y,
        width_m,
        height_m,
    )
    x_values = x_min + (column_values.astype(np.float64) + 0.5) * pixel_width_m
    y_values = y_max - (row_values.astype(np.float64) + 0.5) * pixel_height_m
    longitudes, latitudes = to_latlon.transform(x_values, y_values)
    return {
        "rows": row_values,
        "columns": column_values,
        "longitudes": np.round(np.asarray(longitudes, dtype=np.float64), 6),
        "latitudes": np.round(np.asarray(latitudes, dtype=np.float64), 6),
        "projected_x_m": np.round(x_values, 3),
        "projected_y_m": np.round(y_values, 3),
        "pixel_width_m": pixel_width_m,
        "pixel_height_m": pixel_height_m,
    }


def _pixel_location_from_batch(
    batch: dict[str, Any], index: int, size_x: int, size_y: int
) -> dict[str, Any]:
    row = int(batch["rows"][index])
    column = int(batch["columns"][index])
    pixel_width_m = float(batch["pixel_width_m"])
    pixel_height_m = float(batch["pixel_height_m"])
    return {
        "row": row,
        "column": column,
        # NumPy uses zero-based indices. Keep machine and human ordinals separate.
        "index_base": 0,
        "human_position": {"row": row + 1, "column": column + 1},
        "relative_position": {
            "from_left_fraction": round((column + 0.5) / size_x, 4),
            "from_top_fraction": round((row + 0.5) / size_y, 4),
            "description": "relative to the image width and height; not a compass direction",
        },
        "latitude": float(batch["latitudes"][index]),
        "longitude": float(batch["longitudes"][index]),
        "projected_x_m": float(batch["projected_x_m"][index]),
        "projected_y_m": float(batch["projected_y_m"][index]),
        "pixel_resolution_m": [round(pixel_height_m, 3), round(pixel_width_m, 3)],
        "physical_calibration_available": True,
        "calibration_source": "SEVIR CATALOG projection, bounds, image dimensions, and physical extent",
        "coordinate_reference_system": "WGS84 (derived from event projection)",
    }


def _pixel_latlon(record: dict[str, Any], row: int, column: int) -> dict[str, Any]:
    batch = _pixel_centres_latlon(record, [row], [column])
    return _pixel_location_from_batch(batch, 0, int(record["size_x"]), int(record["size_y"]))


def _features_intersecting_bbox(
    features: tuple[dict[str, Any], ...],
    *,
    south: float,
    west: float,
    north: float,
    east: float,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        feature
        for feature in features
        if not (
            float(feature["bbox"][2]) < west
            or float(feature["bbox"][0]) > east
            or float(feature["bbox"][3]) < south
            or float(feature["bbox"][1]) > north
        )
    )


def _batch_match_boundary_features(
    candidates: tuple[dict[str, Any], ...],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> list[dict[str, Any] | None]:
    """Assign points to a small bbox-filtered polygon set in vector batches."""
    matches: list[dict[str, Any] | None] = [None] * int(latitudes.size)
    unmatched = np.ones(latitudes.size, dtype=bool)
    for feature in candidates:
        min_lon, min_lat, max_lon, max_lat = [float(value) for value in feature["bbox"]]
        candidate_mask = (
            unmatched
            & (longitudes >= min_lon) & (longitudes <= max_lon)
            & (latitudes >= min_lat) & (latitudes <= max_lat)
        )
        candidate_indexes = np.flatnonzero(candidate_mask)
        if candidate_indexes.size == 0:
            continue
        inside = np.zeros(candidate_indexes.size, dtype=bool)
        for ring in feature["polygons"]:
            inside |= _points_in_ring_batch(
                longitudes[candidate_indexes], latitudes[candidate_indexes], ring
            )
        matched_indexes = candidate_indexes[inside]
        for point_index in matched_indexes.tolist():
            matches[point_index] = feature["fields"]
        unmatched[matched_indexes] = False
        if not np.any(unmatched):
            break
    return matches


def _batch_reverse_geocode_us(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    latitude_values = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    longitude_values = np.asarray(longitudes, dtype=np.float64).reshape(-1)
    if latitude_values.size != longitude_values.size:
        raise ValueError("latitude and longitude arrays must have the same length")
    state_path = BOUNDARY_ROOT / "cb_2025_us_state_20m.kmz"
    county_path = BOUNDARY_ROOT / "cb_2025_us_county_20m.kmz"
    if not state_path.exists() or not county_path.exists():
        missing = [state_path.name, county_path.name]
        return ([
            {
                "status": "boundary_data_unavailable",
                "capabilities": [],
                "latitude": float(latitude),
                "longitude": float(longitude),
                "country": None,
                "state": None,
                "state_code": None,
                "state_fips": None,
                "county": None,
                "county_fips": None,
                "required_files": missing,
            }
            for latitude, longitude in zip(latitude_values, longitude_values)
        ], {"candidate_county_count": 0, "candidate_state_count": 0})
    south = float(latitude_values.min())
    west = float(longitude_values.min())
    north = float(latitude_values.max())
    east = float(longitude_values.max())
    county_candidates = _features_intersecting_bbox(
        _load_boundary_kmz(str(county_path)), south=south, west=west, north=north, east=east
    )
    state_candidates = _features_intersecting_bbox(
        _load_boundary_kmz(str(state_path)), south=south, west=west, north=north, east=east
    )
    county_matches = _batch_match_boundary_features(
        county_candidates, latitude_values, longitude_values
    )
    state_matches = _batch_match_boundary_features(
        state_candidates, latitude_values, longitude_values
    )
    results: list[dict[str, Any]] = []
    for latitude, longitude, county, state in zip(
        latitude_values, longitude_values, county_matches, state_matches
    ):
        state_name = (county or {}).get("STATE_NAME") or (state or {}).get("NAME")
        state_code = (county or {}).get("STUSPS") or (state or {}).get("STUSPS")
        state_fips = (county or {}).get("STATEFP") or (state or {}).get("STATEFP")
        status = "ok" if state_name else "outside_us_boundaries"
        results.append({
            "status": status,
            "capabilities": ["offline_state_county_polygon_lookup"],
            "latitude": float(latitude),
            "longitude": float(longitude),
            "country": "United States" if state_name else None,
            "state": state_name,
            "state_code": state_code,
            "state_fips": state_fips,
            "county": county.get("NAME") if county else None,
            "county_fips": county.get("GEOID") if county else None,
            "source": "US Census 2025 cartographic boundary files, 1:20m",
            "provenance": {
                "coordinate_reference_system": "WGS84",
                "administrative_boundary_source": "US Census 2025 cartographic boundaries, 1:20m",
                "point_in_polygon_lookup": bool(state_name),
                "state_lookup_method": "point_in_polygon" if state_name else "not_found",
                "county_lookup_method": "point_in_polygon" if county else "not_found",
            },
            "interpretation_limit": "Polygon lookup only; not a city or street geocoder.",
        })
    return results, {
        "candidate_county_count": len(county_candidates),
        "candidate_state_count": len(state_candidates),
    }


def pixel_box_to_latlon_bounds(
    event_id: str,
    box: list[int] | tuple[int, int, int, int],
) -> dict[str, Any]:
    """Convert an event pixel box to a conservative WGS84 bounding box."""
    record = _catalog_vil_event(event_id)
    if len(box) != 4:
        raise ValueError("box must be [row0, col0, row1, col1]")
    row0, col0, row1, col1 = [int(value) for value in box]
    row0 = max(0, min(int(record["size_y"]) - 1, row0))
    col0 = max(0, min(int(record["size_x"]) - 1, col0))
    row1 = max(row0 + 1, min(int(record["size_y"]), row1))
    col1 = max(col0 + 1, min(int(record["size_x"]), col1))
    corners = [
        _pixel_latlon(record, row, column)
        for row, column in (
            (row0, col0),
            (row0, col1 - 1),
            (row1 - 1, col0),
            (row1 - 1, col1 - 1),
        )
    ]
    return {
        "status": "ok",
        "event_id": event_id,
        "pixel_box": [row0, col0, row1, col1],
        "south": min(item["latitude"] for item in corners),
        "west": min(item["longitude"] for item in corners),
        "north": max(item["latitude"] for item in corners),
        "east": max(item["longitude"] for item in corners),
        "coordinate_reference_system": "WGS84",
    }


def geolocate_pixel_region(
    event_id: str,
    pixels: list[list[int]] | list[tuple[int, int]],
    *,
    max_pixels: int = 5000,
) -> dict[str, Any]:
    """Resolve the administrative coverage of a calibrated raster region.

    ``pixels`` contains zero-based ``[row, column]`` locations belonging to one
    analysis region.  Coverage is calculated from pixel centres against Census
    polygons.  It is intentionally *not* presented as an exact polygon-area
    intersection, so callers cannot silently turn a centre-point lookup into a
    claim about an entire bounding box or county.
    """
    record = _catalog_vil_event(event_id)
    if not isinstance(pixels, list) or not pixels:
        raise ValueError("pixels must contain at least one [row, column] pair")
    if len(pixels) > int(max_pixels):
        raise ValueError(f"region contains {len(pixels)} pixels; maximum is {max_pixels}")

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in pixels:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each pixel must be [row, column]")
        point = (int(item[0]), int(item[1]))
        if point not in seen:
            seen.add(point)
            normalized.append(point)
    points = np.asarray(normalized, dtype=np.int64)
    rows = points[:, 0]
    columns = points[:, 1]
    centroid_row = int(round(float(np.mean(rows))))
    centroid_column = int(round(float(np.mean(columns))))
    all_rows = np.concatenate((rows, np.asarray([centroid_row], dtype=np.int64)))
    all_columns = np.concatenate((columns, np.asarray([centroid_column], dtype=np.int64)))
    transformed = _pixel_centres_latlon(record, all_rows, all_columns)
    total = len(normalized)
    latitudes = np.asarray(transformed["latitudes"][:total], dtype=np.float64)
    longitudes = np.asarray(transformed["longitudes"][:total], dtype=np.float64)
    admin_results, lookup_diagnostics = _batch_reverse_geocode_us(
        np.asarray(transformed["latitudes"], dtype=np.float64),
        np.asarray(transformed["longitudes"], dtype=np.float64),
    )
    coverage: dict[tuple[Any, ...], dict[str, Any]] = {}
    unresolved = 0
    for admin in admin_results[:total]:
        if admin.get("status") not in {"ok", "approximate_boundary_match"}:
            unresolved += 1
        key = (
            admin.get("country"), admin.get("state"), admin.get("state_code"),
            admin.get("county"), admin.get("county_fips"), admin.get("status"),
        )
        entry = coverage.setdefault(key, {
            "country": admin.get("country"),
            "state": admin.get("state"),
            "state_code": admin.get("state_code"),
            "county": admin.get("county"),
            "county_fips": admin.get("county_fips"),
            "lookup_status": admin.get("status"),
            "pixel_count": 0,
        })
        entry["pixel_count"] += 1

    for entry in coverage.values():
        entry["pixel_fraction"] = round(entry["pixel_count"] / total, 6)
    sorted_coverage = sorted(
        coverage.values(), key=lambda item: (-item["pixel_count"], str(item.get("county") or ""))
    )
    centroid = _pixel_location_from_batch(
        transformed, total, int(record["size_x"]), int(record["size_y"])
    )
    center_admin = admin_results[total]
    bounds = {
        "south": float(latitudes.min()),
        "west": float(longitudes.min()),
        "north": float(latitudes.max()),
        "east": float(longitudes.max()),
    }
    return {
        "status": "ok" if unresolved == 0 else "partial",
        "task": "sevir_region_administrative_coverage",
        "event_id": str(event_id),
        "region_scope": "provided_region_pixel_centres",
        "pixel_count": total,
        "pixel_bbox": [int(rows.min()), int(columns.min()), int(rows.max()), int(columns.max())],
        "latlon_bbox": {key: round(float(value), 6) for key, value in bounds.items()},
        "region_centroid": centroid,
        "administrative_center": center_admin,
        "administrative_coverage": sorted_coverage,
        "unresolved_pixel_count": unresolved,
        "provenance": {
            "grid_calibration": "SEVIR CATALOG projection",
            "administrative_boundaries": "US Census 2025 cartographic boundaries, 1:20m",
            "coverage_method": "vectorized_pixel_centres_with_bbox_prefiltered_county_polygons",
            "projection_transform_batches": 1,
            **lookup_diagnostics,
            "matched_county_count": sum(
                1 for item in sorted_coverage if item.get("county") and item.get("pixel_count", 0) > 0
            ),
            "complete_for_provided_pixels": unresolved == 0,
        },
        "interpretation_limit": (
            "Coverage counts raster pixel centres in the supplied region. It does not prove that "
            "the full geographic bounding box or an entire county is affected."
        ),
    }


def latlon_box_to_pixel_box(
    event_id: str,
    bounds: dict[str, float],
) -> dict[str, Any]:
    """Map a WGS84 bounding box to a conservative event pixel box."""
    record = _catalog_vil_event(event_id)
    required = ("south", "west", "north", "east")
    if any(key not in bounds for key in required):
        raise ValueError("bounds must contain south, west, north and east")
    projected_crs = CRS.from_proj4(record["proj"])
    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x_min, _ = to_projected.transform(record["llcrnrlon"], record["llcrnrlat"])
    _, y_max = to_projected.transform(record["urcrnrlon"], record["urcrnrlat"])
    pixel_width = float(record["width_m"]) / int(record["size_x"])
    pixel_height = float(record["height_m"]) / int(record["size_y"])
    points = []
    for latitude in (float(bounds["south"]), float(bounds["north"])):
        for longitude in (float(bounds["west"]), float(bounds["east"])):
            x, y = to_projected.transform(longitude, latitude)
            points.append(
                (
                    int(np.floor((y_max - y) / pixel_height)),
                    int(np.floor((x - x_min) / pixel_width)),
                )
            )
    row0 = max(0, min(int(record["size_y"]), min(point[0] for point in points)))
    col0 = max(0, min(int(record["size_x"]), min(point[1] for point in points)))
    row1 = max(0, min(int(record["size_y"]), max(point[0] for point in points) + 1))
    col1 = max(0, min(int(record["size_x"]), max(point[1] for point in points) + 1))
    return {
        "status": "ok" if row0 < row1 and col0 < col1 else "outside_event_grid",
        "event_id": event_id,
        "pixel_box": [row0, col0, row1, col1],
        "geographic_bounds": {key: float(bounds[key]) for key in required},
        "coordinate_reference_system": "WGS84",
    }


def geolocate_sevir_event(
    event_id: str,
    row: int | None = None,
    column: int | None = None,
) -> dict[str, Any]:
    record = _catalog_vil_event(event_id)
    if (row is None) != (column is None):
        raise ValueError("row and column must be provided together.")
    coordinates = (record["llcrnrlat"], record["llcrnrlon"], record["urcrnrlat"], record["urcrnrlon"])
    if any(value is None for value in coordinates):
        raise ValueError(f"SEVIR event {event_id} has incomplete geographic bounds.")
    min_lat, min_lon, max_lat, max_lon = coordinates
    calibration_available = all(
        record.get(field) not in (None, "")
        for field in ("size_x", "size_y", "width_m", "height_m", "proj")
    )
    result = {
        "status": "ok",
        "task": "sevir_event_geolocation",
        "capabilities": ["event_bounds"]
        + (["projected_grid_calibration"] if calibration_available else [])
        + (["pixel_to_wgs84"] if row is not None and calibration_available else []),
        "provenance": {
            "metadata_source": "SEVIR CATALOG",
            "physical_calibration_available": bool(calibration_available),
            "projection_available": bool(record.get("proj")),
            "administrative_boundary_lookup_performed": False,
        },
        "event_id": record["id"],
        "time_utc": record["time_utc"],
        "event_type": record["event_type"],
        "bounding_box": {
            "south": min_lat,
            "west": min_lon,
            "north": max_lat,
            "east": max_lon,
        },
        "approx_center_latlon": [round((min_lat + max_lat) / 2, 6), round((min_lon + max_lon) / 2, 6)],
        "projection": record["proj"],
        "image_shape": [record["size_y"], record["size_x"]],
        "physical_extent_m": [record["height_m"], record["width_m"]],
        "file_name": record["file_name"],
        "file_index": record["file_index"],
        "metadata_source": "SEVIR CATALOG",
        "interpretation_limit": (
            "The bounding box and projection come from SEVIR CATALOG. Pixel coordinates are "
            "converted with the supplied projection; no administrative reverse geocoding is performed."
        ),
    }
    if row is not None and column is not None:
        result["pixel_location"] = _pixel_latlon(record, int(row), int(column))
    return result
