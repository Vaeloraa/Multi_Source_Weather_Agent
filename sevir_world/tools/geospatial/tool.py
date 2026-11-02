from __future__ import annotations

"""The sole Agent-facing spatial translation capability."""

from typing import Any

import numpy as np

from ...capabilities.base import BoundCapability
from ...capabilities.models import AdministrativeArea, Place
from .._shared import candidate_frame
from .boundaries import boundary as find_boundary
from .boundaries import point_to_location as locate_point
from .contracts import BoundingBox, GeoPoint, GridPoint, Region
from .geocoder import resolve_place as geocode
from .projection import (
    latlon_to_pixel as point_to_grid,
    mask as region_mask,
    pixel_to_latlon as grid_to_point,
    region_bounds as normalize_bounds,
    select_region_value,
)


class Geospatial(BoundCapability):
    """Translate places, WGS84 regions and metadata-described data grids."""

    def resolve_place(self, name: str, country_codes: str = "") -> Place:
        """Resolve a place name or ``latitude,longitude`` text to WGS84.

        Parameters
        ----------
        name : str
            Place name or explicit coordinate pair.
        country_codes : str, optional
            Comma-separated ISO country filter; empty means unrestricted.

        Returns
        -------
        Place
            Selected point and candidates using schema
            ``geospatial.place.candidate@1.0.0``.

        Examples
        --------
        >>> place = geospatial.resolve_place("North Dakota")
        """
        arguments = {"place": str(name), "country_codes": str(country_codes)}
        return self._invoke(
            "geospatial.resolve_place", arguments, geocode,
            lambda raw: Place(
                query=str(raw.get("query") or name), status=str(raw.get("status") or "ok"),
                latitude=float(raw["latitude"]), longitude=float(raw["longitude"]),
                name=str(raw.get("display_name") or name),
                candidates=candidate_frame([
                    {**item, "name": item.get("name") or item.get("display_name") or name}
                    for item in raw.get("candidates", [])
                ]),
            ),
        )

    def point_to_location(self, latitude: float, longitude: float) -> AdministrativeArea:
        """Identify available geographic features containing a WGS84 point.

        Parameters
        ----------
        latitude, longitude : float
            WGS84 decimal degrees.

        Returns
        -------
        AdministrativeArea
            Country, state, county, continent and water-body fields when known.

        Examples
        --------
        >>> location = geospatial.point_to_location(47.5, -100.5)
        """
        arguments = {"latitude": float(latitude), "longitude": float(longitude)}
        return self._invoke(
            "geospatial.point_to_location", arguments, locate_point,
            lambda raw: AdministrativeArea(
                latitude=float(raw["latitude"]), longitude=float(raw["longitude"]),
                status=str(raw.get("status") or "ok"), match_method="point_in_polygon",
                country=raw.get("country"), state=raw.get("state"),
                state_code=raw.get("state_code"), county=raw.get("county"),
                county_fips=raw.get("county_fips"),
                continent=raw.get("continent"), water_body=raw.get("water_body"),
            ),
        )

    def boundary(self, name: str, level: str | None = None) -> Region:
        """Look up a named boundary and its hierarchy metadata.

        Parameters
        ----------
        name : str
            Boundary name, code or identifier.
        level : str, optional
            Boundary level such as ``state`` or ``county``.

        Returns
        -------
        Region
            Canonical polygon, bounding box and hierarchy properties.

        Examples
        --------
        >>> region = geospatial.boundary("North Dakota", level="state")
        """
        arguments = {"name": str(name), "level": level}
        with self._recorder.work("geospatial.boundary", arguments, backend=False):
            return find_boundary(**arguments)

    def region_bounds(self, region: Any) -> BoundingBox:
        """Return canonical WGS84 bounds for a bbox, polygon or Region.

        Parameters
        ----------
        region : Any
            Canonical region, bbox mapping or GeoJSON polygon.

        Returns
        -------
        BoundingBox
            South, west, north and east bounds.

        Examples
        --------
        >>> bounds = geospatial.region_bounds(region)
        """
        with self._recorder.work("geospatial.region_bounds", {"region": region}, backend=False):
            return normalize_bounds(region)

    def pixel_to_latlon(self, data_ref: Any, row: int, column: int) -> GeoPoint:
        """Convert a data grid cell centre to WGS84 using spatial metadata.

        Parameters
        ----------
        data_ref : Any
            DataRef, Event or xarray value carrying ``spatial_metadata``.
        row, column : int
            Zero-based grid indices.

        Returns
        -------
        GeoPoint
            WGS84 and, when applicable, projected coordinates.

        Examples
        --------
        >>> point = geospatial.pixel_to_latlon(observations, 100, 120)
        """
        with self._recorder.work("geospatial.pixel_to_latlon", {"row": row, "column": column}, backend=False):
            return grid_to_point(data_ref, int(row), int(column))

    def latlon_to_pixel(self, data_ref: Any, latitude: float, longitude: float) -> GridPoint:
        """Map a WGS84 point onto a metadata-described data grid.

        Parameters
        ----------
        data_ref : Any
            DataRef, Event or xarray value carrying ``spatial_metadata``.
        latitude, longitude : float
            WGS84 point.

        Returns
        -------
        GridPoint
            Zero-based cell and its actual centre coordinates.

        Examples
        --------
        >>> cell = geospatial.latlon_to_pixel(fields, 47.5, -100.5)
        """
        with self._recorder.work("geospatial.latlon_to_pixel", {"latitude": latitude, "longitude": longitude}, backend=False):
            return point_to_grid(data_ref, float(latitude), float(longitude))

    def mask(self, data_ref: Any, region: Any) -> np.ndarray:
        """Rasterize a WGS84 bbox or polygon onto a data grid.

        Parameters
        ----------
        data_ref : Any
            DataRef or xarray value carrying spatial metadata.
        region : Any
            Canonical Region, bbox mapping or GeoJSON polygon.

        Returns
        -------
        numpy.ndarray
            Read-only boolean array with the target grid shape.

        Examples
        --------
        >>> inside = geospatial.mask(fields, region)
        """
        with self._recorder.work("geospatial.mask", {"region": region}, backend=False):
            return region_mask(data_ref, region)

    def select_region(self, data_ref: Any, region: Any) -> Any:
        """Select a region from an xarray value through the common mask path.

        Parameters
        ----------
        data_ref : Any
            xarray DataArray or Dataset carrying spatial metadata.
        region : Any
            Canonical Region, bbox mapping or GeoJSON polygon.

        Returns
        -------
        Any
            Spatially masked xarray value cropped to a contiguous grid window.
            Spatial metadata is updated for the crop. ``spatial_selection``
            reports the requested and selected bounds; tile intersection does
            not establish observations over the whole requested region.

        Examples
        --------
        >>> local = geospatial.select_region(fields, region)
        """
        if not hasattr(data_ref, "where"):
            raise TypeError("select_region requires an xarray DataArray or Dataset")
        with self._recorder.work("geospatial.select_region", {"region": region}, backend=False):
            return select_region_value(data_ref, region, crop=True)


__all__ = ["Geospatial"]
