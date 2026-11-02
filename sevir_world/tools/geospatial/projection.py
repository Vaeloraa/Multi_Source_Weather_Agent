from __future__ import annotations

"""Metadata-driven grid/projection operations with no data-source branches."""

from functools import lru_cache
from typing import Any, Mapping

import numpy as np
from pyproj import CRS, Geod, Transformer

from .contracts import BoundingBox, GeoPoint, GridPoint, Region, region_from_value


def spatial_metadata(data_ref: Any) -> dict[str, Any]:
    """Read canonical spatial metadata from a DataRef, xarray value or mapping."""

    value = getattr(data_ref, "spatial_metadata", None)
    if value:
        return dict(value)
    attrs = getattr(data_ref, "attrs", None)
    if isinstance(attrs, Mapping) and isinstance(attrs.get("spatial_metadata"), Mapping):
        return dict(attrs["spatial_metadata"])
    if isinstance(data_ref, Mapping):
        candidate = data_ref.get("spatial_metadata", data_ref)
        if isinstance(candidate, Mapping) and candidate.get("grid_type"):
            return dict(candidate)
    coords = getattr(data_ref, "coords", None)
    if coords is not None and "latitude" in coords and "longitude" in coords:
        return {
            "grid_type": "latlon_grid", "crs": "EPSG:4326",
            "latitude": np.asarray(coords["latitude"].values),
            "longitude": np.asarray(coords["longitude"].values),
        }
    raise ValueError("data_ref does not carry spatial_metadata")


def _shape(metadata: Mapping[str, Any]) -> tuple[int, int]:
    shape = tuple(int(item) for item in metadata.get("shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("spatial metadata shape must be [rows, columns]")
    return shape


@lru_cache(maxsize=64)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(source_crs), target_crs, always_xy=True)


def grid_coordinates(data_ref: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return two-dimensional WGS84 latitude/longitude arrays for a grid."""

    metadata = spatial_metadata(data_ref)
    kind = str(metadata.get("grid_type") or "")
    if kind == "latlon_grid":
        latitude = np.asarray(metadata.get("latitude"), dtype=np.float64)
        longitude = np.asarray(metadata.get("longitude"), dtype=np.float64)
        if latitude.ndim == longitude.ndim == 1:
            longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
            return latitude_grid, longitude_grid
        if latitude.shape == longitude.shape and latitude.ndim == 2:
            return latitude, longitude
        raise ValueError("latlon_grid coordinates must be matching 2-D arrays or two 1-D axes")
    if kind != "projected_raster":
        raise ValueError(f"unsupported grid_type {kind!r}")
    rows, columns = _shape(metadata)
    extent = metadata.get("extent") or {}
    try:
        xmin, ymin, xmax, ymax = (float(extent[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("projected_raster metadata requires extent xmin/ymin/xmax/ymax") from error
    x = xmin + (np.arange(columns, dtype=np.float64) + 0.5) * (xmax - xmin) / columns
    y = ymax - (np.arange(rows, dtype=np.float64) + 0.5) * (ymax - ymin) / rows
    xx, yy = np.meshgrid(x, y)
    longitude, latitude = _transformer(str(metadata["crs"]), "EPSG:4326").transform(xx, yy)
    return np.asarray(latitude), np.asarray(longitude)


def pixel_to_latlon(data_ref: Any, row: int, column: int) -> GeoPoint:
    """Convert a zero-based grid cell centre to WGS84."""

    metadata = spatial_metadata(data_ref)
    kind = str(metadata.get("grid_type") or "")
    if kind == "latlon_grid":
        latitudes, longitudes = grid_coordinates(metadata)
        rows, columns = latitudes.shape
        if not 0 <= int(row) < rows or not 0 <= int(column) < columns:
            raise ValueError(f"pixel ({row}, {column}) is outside grid shape ({rows}, {columns})")
        return GeoPoint(float(latitudes[row, column]), float(longitudes[row, column]), int(row), int(column))
    rows, columns = _shape(metadata)
    if not 0 <= int(row) < rows or not 0 <= int(column) < columns:
        raise ValueError(f"pixel ({row}, {column}) is outside grid shape ({rows}, {columns})")
    extent = metadata["extent"]
    x = float(extent["xmin"]) + (int(column) + 0.5) * (float(extent["xmax"]) - float(extent["xmin"])) / columns
    y = float(extent["ymax"]) - (int(row) + 0.5) * (float(extent["ymax"]) - float(extent["ymin"])) / rows
    longitude, latitude = _transformer(str(metadata["crs"]), "EPSG:4326").transform(x, y)
    return GeoPoint(float(latitude), float(longitude), int(row), int(column), x, y, str(metadata["crs"]))


def latlon_to_pixel(data_ref: Any, latitude: float, longitude: float) -> GridPoint:
    """Map a WGS84 point to the nearest/containing grid cell."""

    latitude, longitude = float(latitude), float(longitude)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside WGS84")
    metadata = spatial_metadata(data_ref)
    if metadata.get("grid_type") == "latlon_grid":
        latitudes, longitudes = grid_coordinates(metadata)
        distance = (latitudes - latitude) ** 2 + (((longitudes - longitude + 180) % 360) - 180) ** 2
        row, column = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
        return GridPoint(int(row), int(column), float(latitudes[row, column]), float(longitudes[row, column]))
    rows, columns = _shape(metadata)
    x, y = _transformer("EPSG:4326", str(metadata["crs"])).transform(longitude, latitude)
    extent = metadata["extent"]
    column = int(np.floor((x - float(extent["xmin"])) / (float(extent["xmax"]) - float(extent["xmin"])) * columns))
    row = int(np.floor((float(extent["ymax"]) - y) / (float(extent["ymax"]) - float(extent["ymin"])) * rows))
    if not 0 <= row < rows or not 0 <= column < columns:
        raise ValueError("point is outside the grid")
    centre = pixel_to_latlon(metadata, row, column)
    return GridPoint(row, column, centre.latitude, centre.longitude)


def region_bounds(region: Any) -> BoundingBox:
    return region_from_value(region).bbox


def bounding_boxes_intersect(first: Any, second: Any) -> bool:
    """Return whether two canonical WGS84 bounding boxes intersect."""

    left = region_bounds(first)
    right = region_bounds(second)
    return not (
        left.east < right.west
        or left.west > right.east
        or left.north < right.south
        or left.south > right.north
    )


def bounding_box_center(region: Any) -> GeoPoint:
    """Return the WGS84 centre of a canonical region or bounding box."""

    box = region_bounds(region)
    return GeoPoint(
        latitude=(box.south + box.north) / 2.0,
        longitude=(box.west + box.east) / 2.0,
    )


def geodesic_distance_to_bounds_km(
    latitude: float, longitude: float, region: Any
) -> float:
    """Return WGS84 distance from a point to the nearest point of bounds."""

    latitude, longitude = float(latitude), float(longitude)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside WGS84")
    box = region_bounds(region)
    closest_latitude = min(max(latitude, box.south), box.north)
    closest_longitude = min(max(longitude, box.west), box.east)
    _, _, distance_metres = Geod(ellps="WGS84").inv(
        longitude, latitude, closest_longitude, closest_latitude
    )
    return float(distance_metres) / 1000.0


def _points_in_polygon(longitudes: np.ndarray, latitudes: np.ndarray, polygon: tuple[tuple[float, float], ...]) -> np.ndarray:
    result = np.zeros(longitudes.shape, dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crosses = (y1 > latitudes) != (y2 > latitudes)
        intersection = (x2 - x1) * (latitudes - y1) / ((y2 - y1) or 1e-12) + x1
        result ^= crosses & (longitudes < intersection)
        x1, y1 = x2, y2
    return result


def mask(data_ref: Any, region: Any) -> np.ndarray:
    """Rasterize a WGS84 bbox/polygon against any supported target grid."""

    normalized: Region = region_from_value(region)
    latitudes, longitudes = grid_coordinates(data_ref)
    box = normalized.bbox
    result = (
        (latitudes >= box.south) & (latitudes <= box.north)
        & (longitudes >= box.west) & (longitudes <= box.east)
    )
    if normalized.polygon is not None:
        result &= _points_in_polygon(longitudes, latitudes, normalized.polygon.coordinates)
    result.setflags(write=False)
    return result


def select_region_value(data_ref: Any, region: Any, *, crop: bool = False) -> Any:
    """Mask an xarray grid and retain truthful region and grid metadata.

    Coverage is deliberately unproven: an intersecting source tile and a
    successful pixel read cannot establish observations over a whole region.
    Cropping uses a contiguous window so projected grid transforms stay valid.
    """
    import xarray as xr

    if not isinstance(data_ref, (xr.DataArray, xr.Dataset)):
        raise TypeError("select_region requires an xarray DataArray or Dataset")
    normalized = region_from_value(region)
    selected = mask(data_ref, normalized)
    dims = ("latitude", "longitude") if {"latitude", "longitude"}.issubset(data_ref.dims) else ("row", "column")
    if tuple(data_ref.sizes[name] for name in dims) != selected.shape:
        raise ValueError("spatial metadata grid shape disagrees with the data")
    result = data_ref.where(xr.DataArray(selected, dims=dims))
    result.attrs = dict(data_ref.attrs)
    latitudes, longitudes = grid_coordinates(data_ref)
    count = int(np.count_nonzero(selected))
    result.attrs["spatial_selection"] = {
        "requested_bounds": normalized.bbox.as_dict(),
        "region_name": normalized.name,
        "geometry": "polygon" if normalized.polygon is not None else "bounding_box",
        "selection": "grid_cell_centres_inside_region",
        "selected_grid_cells": count,
        "source_grid_cells": int(selected.size),
        "selected_cell_centre_bounds": {
            "south": float(np.min(latitudes[selected])), "west": float(np.min(longitudes[selected])),
            "north": float(np.max(latitudes[selected])), "east": float(np.max(longitudes[selected])),
        } if count else None,
        "region_coverage": "intersection_only" if count else "no_grid_cells",
        "full_region_coverage_proven": False,
        "coverage_limit": "Tile intersection does not establish complete regional observation coverage.",
    }
    if crop:
        rows, columns = np.nonzero(selected)
        row_start, row_end = (int(rows.min()), int(rows.max()) + 1) if count else (0, 0)
        col_start, col_end = (int(columns.min()), int(columns.max()) + 1) if count else (0, 0)
        result = result.isel({dims[0]: slice(row_start, row_end), dims[1]: slice(col_start, col_end)})
        metadata = dict(spatial_metadata(data_ref))
        if metadata["grid_type"] == "projected_raster":
            extent = metadata["extent"]
            height, width = selected.shape
            dx = (extent["xmax"] - extent["xmin"]) / width
            dy = (extent["ymax"] - extent["ymin"]) / height
            metadata.update(shape=[row_end - row_start, col_end - col_start], extent={
                "xmin": extent["xmin"] + col_start * dx, "xmax": extent["xmin"] + col_end * dx,
                "ymax": extent["ymax"] - row_start * dy, "ymin": extent["ymax"] - row_end * dy,
            })
            metadata.pop("bounds", None)  # Original source bounds no longer describe the crop.
        else:
            metadata.update(latitude=latitudes[row_start:row_end, col_start:col_end],
                            longitude=longitudes[row_start:row_end, col_start:col_end])
        result.attrs["spatial_metadata"] = metadata
    return result


def raster_region_bounds(data_ref: Any, region: Any) -> BoundingBox:
    selected = mask(data_ref, region)
    if not np.any(selected):
        raise ValueError("region does not intersect the grid")
    latitudes, longitudes = grid_coordinates(data_ref)
    return BoundingBox(
        float(np.min(latitudes[selected])), float(np.min(longitudes[selected])),
        float(np.max(latitudes[selected])), float(np.max(longitudes[selected])),
    )


__all__ = [
    "bounding_box_center", "bounding_boxes_intersect",
    "geodesic_distance_to_bounds_km", "grid_coordinates", "latlon_to_pixel",
    "mask", "pixel_to_latlon", "raster_region_bounds", "region_bounds",
    "spatial_metadata", "select_region_value",
]
