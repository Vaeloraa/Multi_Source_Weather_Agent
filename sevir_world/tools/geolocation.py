from __future__ import annotations

from typing import Any

from ..backends import geospatial

from .base import WorldComponent, resolve_event_id


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
        """Convert one zero-based raster pixel to a georeferenced point.

        Returns a dictionary, not a tuple.  Read ``result['latitude']`` and
        ``result['longitude']``; ``row`` and ``column`` are zero-based NumPy
        indices.  For a user-facing ordinal use ``result['human_position']``
        (so row=39 means the 40th row); the result also contains row, column,
        and ``relative_position`` (fractions from the left/top edge, which is
        usually more intuitive for a non-technical user),
        projected_x_m, projected_y_m, pixel_resolution_m,
        physical_calibration_available, calibration_source and
        coordinate_reference_system.  A real compass direction requires two
        distinct calls/endpoints.  Do not write ``lat, lon =
        pixel_to_latlon(...)`` because iterating the dictionary yields keys.
        """
        location = self.locate(event_id, row=row, column=column)
        return location["pixel_location"]

    def reverse_geocode(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Resolve a WGS84 point to offline Census state and county polygons."""
        return self._call(
            "geolocator.reverse_geocode",
            {"latitude": float(latitude), "longitude": float(longitude)},
            geospatial.reverse_geocode_us,
        )

    def resolve_region(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Return offline administrative names for a user-supplied WGS84 point."""
        return self.reverse_geocode(float(latitude), float(longitude))

    def resolve_place(self, place: str, country_codes: str = "us") -> dict[str, Any]:
        """Resolve a place name or explicit ``latitude,longitude`` text to WGS84."""
        return self._call(
            "geolocator.resolve_place",
            {"place": str(place), "country_codes": str(country_codes)},
            geospatial.resolve_place,
        )

    def locate_admin(self, event_id: Any, row: int = 192, column: int = 192) -> dict[str, Any]:
        """Resolve an event pixel to WGS84 and offline state/county."""
        event_id = resolve_event_id(event_id)
        pixel = self.pixel_to_latlon(event_id, row, column)
        admin = self.reverse_geocode(pixel["latitude"], pixel["longitude"])
        return {
            "status": "ok" if admin.get("status") == "ok" else admin.get("status"),
            "task": "sevir_pixel_administrative_location",
            "capabilities": ["pixel_to_wgs84", "offline_state_county_polygon_lookup"],
            "event_id": event_id,
            "pixel_location": pixel,
            "administrative_area": admin,
            "provenance": {
                "grid_calibration": pixel.get("calibration_source"),
                "administrative_boundaries": admin.get("source"),
                "location_is_image_derived_guess": False,
            },
        }

    def region_administrative_coverage(
        self, event_id: Any, pixels: list[list[int]]
    ) -> dict[str, Any]:
        """Resolve county/state coverage for all pixels in one analysis region.

        The result distinguishes a region-wide pixel-centre coverage calculation
        from the separate administrative lookup for its centroid.
        """
        event_id = resolve_event_id(event_id)
        return self._call(
            "geolocator.region_administrative_coverage",
            {"event_id": event_id, "pixels": pixels},
            geospatial.geolocate_pixel_region,
        )

    def pixel_box_to_latlon_bounds(
        self, event_id: Any, box: list[int]
    ) -> dict[str, Any]:
        """Convert an event pixel box to a conservative WGS84 bounding box."""
        event_id = resolve_event_id(event_id)
        return self._call(
            "geolocator.pixel_box_to_latlon_bounds",
            {"event_id": event_id, "box": box},
            geospatial.pixel_box_to_latlon_bounds,
        )

    def latlon_box_to_pixel_box(
        self, event_id: Any, bounds: dict[str, float]
    ) -> dict[str, Any]:
        """Map a WGS84 bounding box to the corresponding event pixel box."""
        event_id = resolve_event_id(event_id)
        return self._call(
            "geolocator.latlon_box_to_pixel_box",
            {"event_id": event_id, "bounds": bounds},
            geospatial.latlon_box_to_pixel_box,
        )
