from __future__ import annotations

"""Public geolocation tool with request-bound SEVIR coordinate conversion."""

from typing import Any

import pandas as pd

from ...capabilities.base import BoundCapability
from ...capabilities.models import (
    AdministrativeArea,
    Location,
    PixelLocation,
    Place,
    RegionCoverage,
)
from .._shared import candidate_frame, pixel_value


class Geolocator(BoundCapability):
    """
    Geolocator: public API for request-bound coordinate and place operations.

    This wrapper exposes a clean interface to:
      • Read the calibrated WGS84 footprint of a SEVIR event
      • Convert raster pixels to geographic coordinates
      • Resolve place names and administrative areas
      • Describe the geographic coverage of Agent-selected pixels

    Parameters
    ----------
    recorder : callable
        Runtime callback that records public tool calls and cache hits.
    binding : RequestBinding
        Request identity constructed before the Agent starts. Event, upload,
        location and requested time are already bound to this object.
    """

    # ------------------------------------------------------------------
    # Event-grid discovery
    # ------------------------------------------------------------------
    def event(self, subject_key: str | None = None) -> Location:
        """
        Return the calibrated geographic footprint of one SEVIR event.

        Parameters
        ----------
        subject_key : str, optional
            Exact key returned by ``sevir.search()['subject_key']``. When
            omitted, use the subject bound to this request.

        Returns
        -------
        Location
            Flat object with ``subject_key``, ``event_id``, ``latitude``, ``longitude``,
            ``south``, ``west``, ``north``, ``east``, ``row_count``,
            ``column_count`` and ``crs``.

        Examples
        --------
        >>> location = geolocator.event()
        >>> latitude, longitude = location.latitude, location.longitude
        >>> rows, columns = location.row_count, location.column_count
        """
        from .Geospatial import geolocate_sevir_event

        value = self._subject_key(subject_key)

        def convert(raw: dict[str, Any]) -> Location:
            """Convert backend event geolocation data to ``Location``.

            Parameters
            ----------
            raw : dict
                Event location and administrative-area fields.

            Returns
            -------
            Location
                Public typed location for the request-bound event.
            """
            bounds = raw["bounding_box"]
            return Location(
                subject_key=str(raw.get("subject_key") or value).lower(),
                event_id=str(raw["event_id"]).upper(),
                latitude=float(raw["approx_center_latlon"][0]),
                longitude=float(raw["approx_center_latlon"][1]),
                south=float(bounds["south"]), west=float(bounds["west"]),
                north=float(bounds["north"]), east=float(bounds["east"]),
                row_count=int(raw["image_shape"][0]),
                column_count=int(raw["image_shape"][1]),
                crs=str(raw.get("projection") or ""),
            )

        return self._invoke(
            "geolocator.event", {"event_id": value}, geolocate_sevir_event, convert,
            public_arguments={"subject_key": value},
        )

    def pixel(
        self,
        row: int,
        column: int,
        subject_key: str | None = None,
    ) -> PixelLocation:
        """
        Convert one zero-based event raster pixel to WGS84 coordinates.

        Parameters
        ----------
        row : int
            Zero-based raster row.
        column : int
            Zero-based raster column.
        subject_key : str, optional
            Exact key returned by ``sevir.search``. When omitted, use the
            request-bound SEVIR subject.

        Returns
        -------
        PixelLocation
            Object with fields ``row``, ``column``, ``latitude``, ``longitude``,
            ``projected_x`` and ``projected_y``.

        Examples
        --------
        >>> point = geolocator.pixel(120, 180)
        >>> point.latitude, point.longitude
        """
        from .Geospatial import geolocate_sevir_event

        arguments = {
            "event_id": self._subject_key(subject_key),
            "row": int(row),
            "column": int(column),
        }
        return self._invoke(
            "geolocator.pixel",
            arguments,
            geolocate_sevir_event,
            lambda raw: pixel_value(raw["pixel_location"]),
            public_arguments={
                "row": int(row),
                "column": int(column),
                "subject_key": arguments["event_id"],
            },
        )

    # ------------------------------------------------------------------
    # Place and administrative lookup
    # ------------------------------------------------------------------
    def place(self, name: str, country_codes: str = "us") -> Place:
        """
        Resolve a place name or ``latitude,longitude`` text to a WGS84 point.

        Parameters
        ----------
        name : str
            Place name or numeric ``latitude,longitude`` pair.
        country_codes : str, default "us"
            Country filter passed to the configured place-name geocoder.

        Returns
        -------
        Place
            Object with fields ``query``, ``status``, ``latitude``, ``longitude``, ``name``
            and ``candidates``. Candidates are returned as a pandas DataFrame
            with columns ``name``, ``latitude``, ``longitude``, ``category``
            and ``place_type``.

        Examples
        --------
        >>> springfield = geolocator.place("Springfield, Missouri")
        >>> springfield.latitude, springfield.longitude
        """
        from .Geospatial import resolve_place

        arguments = {"place": str(name), "country_codes": str(country_codes)}
        return self._invoke(
            "geolocator.place",
            arguments,
            resolve_place,
            lambda raw: Place(
                query=str(raw.get("query") or name),
                status=str(raw.get("status") or "ok"),
                latitude=float(raw["latitude"]),
                longitude=float(raw["longitude"]),
                name=str(raw.get("display_name") or name),
                candidates=candidate_frame([
                    {"name": str(item.get("display_name") or name),
                     "latitude": float(item["latitude"]),
                     "longitude": float(item["longitude"]),
                     "category": str(item["category"]) if item.get("category") is not None else None,
                     "place_type": str(item["type"]) if item.get("type") is not None else None}
                    for item in raw.get("candidates", [])
                ]),
            ),
            public_arguments={"name": str(name), "country_codes": str(country_codes)},
        )

    def administrative_area(
        self,
        latitude: float,
        longitude: float,
    ) -> AdministrativeArea:
        """
        Resolve a WGS84 point against bundled US state and county polygons.

        Parameters
        ----------
        latitude, longitude : float
            WGS84 point in decimal degrees.

        Returns
        -------
        AdministrativeArea
            Object with fields ``latitude``, ``longitude``, ``status``,
            ``match_method``, ``country``,
            ``state``, ``state_code``, ``county`` and ``county_fips``.
            Administrative fields may be None outside available boundaries.

        Examples
        --------
        >>> area = geolocator.administrative_area(38.0, -98.0)
        >>> area.state, area.county
        """
        from .Geospatial import reverse_geocode_us

        arguments = {"latitude": float(latitude), "longitude": float(longitude)}
        return self._invoke(
            "geolocator.administrative_area",
            arguments,
            reverse_geocode_us,
            lambda raw: AdministrativeArea(
                latitude=float(raw["latitude"]),
                longitude=float(raw["longitude"]),
                status=str(raw.get("status") or "ok"),
                match_method=str(raw.get("match_method") or raw.get("method") or ""),
                country=raw.get("country"),
                state=raw.get("state"),
                state_code=raw.get("state_code"),
                county=raw.get("county"),
                county_fips=raw.get("county_fips"),
            ),
        )

    def region(
        self,
        pixels: list[list[int]],
        subject_key: str | None = None,
    ) -> RegionCoverage:
        """
        Map Agent-selected pixels to geographic and administrative coverage.

        Parameters
        ----------
        pixels : list[list[int]]
            Non-empty zero-based ``[row, column]`` pixel locations belonging to
            the region being analysed.
        subject_key : str, optional
            Exact key returned by ``sevir.search``. When omitted, use the
            request-bound SEVIR subject.

        Returns
        -------
        RegionCoverage
            Flat object with ``subject_key``, ``event_id``, pixel bounds, geographic bounds, centroid
            coordinates and an ``areas`` pandas DataFrame.

        Examples
        --------
        >>> coverage = geolocator.region([[100, 100], [101, 100]])
        >>> coverage.south, coverage.west, coverage.north, coverage.east
        >>> coverage.areas
        """
        from .Geospatial import geolocate_pixel_region

        arguments = {"event_id": self._subject_key(subject_key), "pixels": pixels}

        def convert(raw: dict[str, Any]) -> RegionCoverage:
            """Convert a backend region result to ``RegionCoverage``.

            Parameters
            ----------
            raw : dict
                Bounding box, corner coordinates and administrative coverage.

            Returns
            -------
            RegionCoverage
                Public typed description of the selected pixel region.
            """
            bounds = raw["latlon_bbox"]
            pixel_box = tuple(int(item) for item in raw["pixel_bbox"])
            centroid = pixel_value(raw["region_centroid"])
            return RegionCoverage(
                subject_key=str(raw.get("subject_key") or self._subject_key(subject_key)).lower(),
                event_id=str(raw["event_id"]).upper(),
                pixel_count=int(raw["pixel_count"]),
                min_row=pixel_box[0], min_column=pixel_box[1],
                max_row=pixel_box[2], max_column=pixel_box[3],
                latitude=centroid.latitude, longitude=centroid.longitude,
                south=float(bounds["south"]), west=float(bounds["west"]),
                north=float(bounds["north"]), east=float(bounds["east"]),
                unresolved_pixel_count=int(raw.get("unresolved_pixel_count", 0)),
                areas=pd.DataFrame(raw.get("administrative_coverage", [])),
            )

        return self._invoke(
            "geolocator.region", arguments, geolocate_pixel_region, convert,
            public_arguments={
                "pixels": pixels,
                "subject_key": arguments["event_id"],
            },
        )
