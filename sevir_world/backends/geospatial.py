from __future__ import annotations

import base64
import binascii
import csv
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..repositories.config import *
from ..repositories.catalog import _catalog_vil_event

import h5py
import numpy as np
from pyproj import CRS, Transformer


def _pixel_latlon(record: dict[str, Any], row: int, column: int) -> dict[str, Any]:
    size_x = record["size_x"]
    size_y = record["size_y"]
    width_m = record["width_m"]
    height_m = record["height_m"]
    proj = record["proj"]
    if not all((size_x, size_y, width_m, height_m, proj)):
        raise ValueError(f"SEVIR event {record['id']} has incomplete projection metadata.")
    if not 0 <= row < size_y or not 0 <= column < size_x:
        raise ValueError(
            f"Pixel (row={row}, column={column}) is outside image shape ({size_y}, {size_x})."
        )

    projected_crs = CRS.from_proj4(proj)
    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_latlon = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    x_min, _ = to_projected.transform(record["llcrnrlon"], record["llcrnrlat"])
    _, y_max = to_projected.transform(record["urcrnrlon"], record["urcrnrlat"])
    pixel_width_m = width_m / size_x
    pixel_height_m = height_m / size_y
    x = x_min + (column + 0.5) * pixel_width_m
    y = y_max - (row + 0.5) * pixel_height_m
    longitude, latitude = to_latlon.transform(x, y)
    return {
        "row": row,
        "column": column,
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "projected_x_m": round(float(x), 3),
        "projected_y_m": round(float(y), 3),
        "pixel_resolution_m": [round(float(pixel_height_m), 3), round(float(pixel_width_m), 3)],
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
    result = {
        "status": "ok",
        "task": "sevir_event_geolocation",
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
