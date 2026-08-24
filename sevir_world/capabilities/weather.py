from __future__ import annotations

"""The complete public capability surface available to generated programs.

Every instance is bound to one request when it is constructed.  Public calls
return small dataclasses, NumPy arrays, lists, tuples, and dictionaries; the
backend's transport/status envelopes never cross this boundary.
"""

from typing import Any

import numpy as np

from .base import BoundCapability, RequestBinding
from .models import (
    AdministrativeArea,
    Bounds,
    CatalogEvent,
    EnvironmentFields,
    Event,
    Forecast,
    Location,
    ObservationCube,
    PixelLocation,
    Place,
    RegionCoverage,
    UploadMatch,
)


def _bounds(value: Any) -> Bounds | None:
    if not isinstance(value, dict):
        return None
    required = ("south", "west", "north", "east")
    if any(value.get(key) is None for key in required):
        return None
    return Bounds(*(float(value[key]) for key in required))


def _number(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _catalog_value(record: dict[str, Any]) -> CatalogEvent:
    event_id = str(record.get("id") or record.get("event_id") or "").upper()
    bounds = _bounds(record.get("bounding_box") or record.get("latlon_bbox"))
    missing = record.get("pct_missing")
    if missing is not None:
        missing = float(missing) / 100.0
    known = {
        "id", "event_id", "time_utc", "event_type", "data_min", "data_max",
        "pct_missing", "bounding_box", "latlon_bbox", "distance_km",
        "local_hdf5_available", "local_data_available",
    }
    return CatalogEvent(
        id=event_id,
        time_utc=str(record.get("time_utc") or ""),
        event_type=str(record.get("event_type") or "unlabeled"),
        data_min=_number(record.get("data_min")),
        data_max=_number(record.get("data_max")),
        missing_fraction=missing,
        bounds=bounds,
        distance_km=_number(record.get("distance_km", record.get("distance_to_footprint_km"))),
        local_data=bool(
            record.get("local_hdf5_available", record.get("local_data_available", False))
        ),
        metadata={key: value for key, value in record.items() if key not in known},
    )


def _event_value(raw: dict[str, Any]) -> Event:
    record = dict(raw["record"])
    offsets = np.asarray(record.pop("minute_offsets", []) or [], dtype=np.int32)
    offsets.setflags(write=False)
    return Event(
        id=str(record.pop("id")).upper(),
        time_utc=str(record.pop("time_utc", "")),
        event_type=str(record.pop("event_type", "unlabeled")),
        frame_times_utc=tuple(str(item) for item in record.pop("frame_times_utc", [])),
        minute_offsets=offsets,
        bounds=_bounds(record.pop("bounding_box", None)),
        metadata=record,
    )


def _observation_value(raw: dict[str, Any]) -> ObservationCube:
    values = np.ascontiguousarray(raw.get("values", raw.get("frames")), dtype=np.float32)
    values.setflags(write=False)
    minutes = np.asarray(raw.get("relative_minutes") or [], dtype=np.int32)
    minutes.setflags(write=False)
    return ObservationCube(
        event_id=str(raw.get("event_id") or "").upper(),
        values=values,
        minutes=minutes,
        missing_value=int(raw.get("missing_value", 255)),
        dimensions=tuple(raw.get("dimensions") or ("time", "row", "column")),
    )


def _pixel_value(value: dict[str, Any]) -> PixelLocation:
    return PixelLocation(
        row=int(value["row"]),
        column=int(value["column"]),
        latitude=float(value["latitude"]),
        longitude=float(value["longitude"]),
        projected_x=_number(value.get("projected_x_m")),
        projected_y=_number(value.get("projected_y_m")),
    )


def _forecast_value(raw: dict[str, Any]) -> Forecast:
    mean = np.ascontiguousarray(raw["forecast_frames"], dtype=np.float32)
    mean.setflags(write=False)
    member_records = raw.get("ensemble", {}).get("members", [])
    if member_records:
        members = np.ascontiguousarray(
            np.stack([np.asarray(item["frames"], dtype=np.float32) for item in member_records])
        )
    else:
        members = mean[None, ...]
    members.setflags(write=False)
    leads = np.asarray(
        raw.get("forecast_relative_minutes") or raw.get("lead_minutes") or [],
        dtype=np.int32,
    )
    leads.setflags(write=False)
    truth_record = raw.get("ground_truth") if isinstance(raw.get("ground_truth"), dict) else {}
    truth: np.ndarray | None = None
    if truth_record.get("available") and np.asarray(truth_record.get("frames", [])).size:
        truth = np.ascontiguousarray(truth_record["frames"], dtype=np.float32)
        truth.setflags(write=False)
    seeds = tuple(
        int(item.get("seed", index + 1))
        for index, item in enumerate(member_records)
        if item.get("seed") is not None
    )
    return Forecast(
        subject_id=str(raw.get("event_id") or raw.get("upload_id") or ""),
        model=str(raw.get("model_name") or "cascast"),
        mean=mean,
        members=members,
        lead_minutes=leads,
        ground_truth=truth,
        member_seeds=seeds,
        cache_hit=bool(raw.get("cache_hit", False)),
    )


class Sevir(BoundCapability):
    """SEVIR catalog and observation access bound to the current request."""

    def event(self, event_id: str | None = None) -> Event:
        """Return one event's metadata and time axis.

        Args: event_id overrides the request-bound event only when supplied.
        Returns: Event with direct attributes such as ``event.time_utc``.
        Example: ``event = sevir.event(); event.minute_offsets[-1]``
        """
        from ..data_access.catalog import get_sevir_event_metadata

        value = self._event_id(event_id)
        return self._invoke("sevir.event", {"event_id": value}, get_sevir_event_metadata, _event_value)

    def search(
        self,
        query: str = "",
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        limit: int = 10,
    ) -> list[CatalogEvent]:
        """Search catalog metadata without loading pixels.

        Args: filters are optional; limit is capped by the backend.
        Returns: list[CatalogEvent], ready for sorting and filtering.
        Example: ``events = sevir.search(year=2019, month=6); events[0].id``
        """
        from ..data_access.catalog import search_sevir_catalog

        arguments = {"query": query, "year": year, "month": month, "event_type": event_type, "limit": int(limit)}
        return self._invoke(
            "sevir.search", arguments, search_sevir_catalog,
            lambda raw: [_catalog_value(item) for item in raw.get("records", [])],
        )

    def nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100.0,
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        limit: int = 50,
    ) -> list[CatalogEvent]:
        """Find catalog events whose footprints are near a WGS84 point.

        Args: latitude/longitude locate the point; remaining values filter results.
        Returns: distance-ordered list[CatalogEvent] containing metadata only.
        Example: ``near = sevir.nearby(38.0, -98.0, radius_km=200)``
        """
        from ..data_access.catalog import search_sevir_catalog_by_region

        arguments = {
            "latitude": float(latitude), "longitude": float(longitude),
            "radius_km": float(radius_km), "year": year, "month": month,
            "event_type": event_type, "limit": int(limit),
        }
        return self._invoke(
            "sevir.nearby", arguments, search_sevir_catalog_by_region,
            lambda raw: [_catalog_value(item) for item in raw.get("records", [])],
        )

    def observations(
        self,
        event_id: str | None = None,
        start_frame: int = 0,
        end_frame: int = 49,
    ) -> ObservationCube:
        """Load a VIL cube whose axes are time, row, and column.

        Args: the frame interval is a normal half-open Python slice.
        Returns: ObservationCube; use ``cube.values`` directly with NumPy. Value 255 is missing.
        Example: ``cube = sevir.observations(end_frame=13); np.nanmax(np.where(cube.values == 255, np.nan, cube.values))``
        """
        from ..data_access.catalog import load_sevir_vil_event

        arguments = {
            "event_id": self._event_id(event_id),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
        }
        return self._invoke("sevir.observations", arguments, load_sevir_vil_event, _observation_value)

    def match_upload(self, upload_id: str | None = None) -> UploadMatch:
        """Match a bound upload to a verified catalog event when possible.

        Args: upload_id overrides the request-bound upload only when supplied.
        Returns: UploadMatch; ``event_id`` is None when no unique match exists.
        Example: ``match = sevir.match_upload(); match.event_id``
        """
        from ..data_access.uploads import match_uploaded_event

        value = self._upload_id(upload_id)

        def convert(raw: dict[str, Any]) -> UploadMatch:
            match = raw.get("event_match") if isinstance(raw.get("event_match"), dict) else {}
            return UploadMatch(
                upload_id=value,
                event_id=str(match.get("event_id") or "").upper() or None,
                confidence=_number(match.get("confidence")),
                method=str(match.get("method") or match.get("status") or "unmatched"),
            )

        return self._invoke("sevir.match_upload", {"upload_id": value}, match_uploaded_event, convert)


class Geolocator(BoundCapability):
    """Coordinate conversion and place lookup bound to the current request."""

    def event(self, event_id: str | None = None) -> Location:
        """Return the calibrated footprint of one SEVIR event.

        Args: event_id overrides the request-bound event only when supplied.
        Returns: Location with WGS84 bounds, center, shape, and projection.
        Example: ``location = geolocator.event(); location.center``
        """
        from ..backends.geospatial import geolocate_sevir_event

        value = self._event_id(event_id)

        def convert(raw: dict[str, Any]) -> Location:
            bounds = _bounds(raw["bounding_box"])
            assert bounds is not None
            return Location(
                event_id=str(raw["event_id"]).upper(),
                center=tuple(float(item) for item in raw["approx_center_latlon"]),
                bounds=bounds,
                image_shape=tuple(int(item) for item in raw["image_shape"]),
                projection=str(raw.get("projection") or ""),
            )

        return self._invoke("geolocator.event", {"event_id": value}, geolocate_sevir_event, convert)

    def pixel(self, row: int, column: int, event_id: str | None = None) -> PixelLocation:
        """Convert one zero-based raster pixel to WGS84.

        Args: row and column are zero-based; event_id is normally omitted.
        Returns: PixelLocation with geographic and projected coordinates.
        Example: ``point = geolocator.pixel(120, 180); point.latitude``
        """
        from ..backends.geospatial import geolocate_sevir_event

        arguments = {"event_id": self._event_id(event_id), "row": int(row), "column": int(column)}
        return self._invoke(
            "geolocator.pixel", arguments, geolocate_sevir_event,
            lambda raw: _pixel_value(raw["pixel_location"]),
        )

    def place(self, name: str, country_codes: str = "us") -> Place:
        """Resolve a place name or ``latitude,longitude`` text.

        Args: name is the query and country_codes narrows network geocoding.
        Returns: Place containing the selected point and any candidates.
        Example: ``springfield = geolocator.place('Springfield, Missouri')``
        """
        from ..backends.geospatial import resolve_place

        arguments = {"place": str(name), "country_codes": str(country_codes)}
        return self._invoke(
            "geolocator.place", arguments, resolve_place,
            lambda raw: Place(
                query=str(raw.get("query") or name),
                latitude=float(raw["latitude"]),
                longitude=float(raw["longitude"]),
                name=str(raw.get("display_name") or name),
                candidates=tuple(dict(item) for item in raw.get("candidates", [])),
            ),
        )

    def administrative_area(self, latitude: float, longitude: float) -> AdministrativeArea:
        """Resolve a WGS84 point against bundled US state/county polygons.

        Args: latitude and longitude are decimal degrees.
        Returns: AdministrativeArea; absent fields are None.
        Example: ``area = geolocator.administrative_area(38.0, -98.0); area.state``
        """
        from ..backends.geospatial import reverse_geocode_us

        arguments = {"latitude": float(latitude), "longitude": float(longitude)}
        return self._invoke(
            "geolocator.administrative_area", arguments, reverse_geocode_us,
            lambda raw: AdministrativeArea(
                latitude=float(raw["latitude"]), longitude=float(raw["longitude"]),
                country=raw.get("country"), state=raw.get("state"),
                state_code=raw.get("state_code"), county=raw.get("county"),
                county_fips=raw.get("county_fips"),
            ),
        )

    def region(self, pixels: list[list[int]], event_id: str | None = None) -> RegionCoverage:
        """Map Agent-selected pixels to their geographic and administrative coverage.

        Args: pixels is a non-empty list of zero-based ``[row, column]`` pairs.
        Returns: RegionCoverage for exactly those pixel centres.
        Example: ``coverage = geolocator.region([[100, 100], [101, 100]])``
        """
        from ..backends.geospatial import geolocate_pixel_region

        arguments = {"event_id": self._event_id(event_id), "pixels": pixels}

        def convert(raw: dict[str, Any]) -> RegionCoverage:
            bounds = _bounds(raw["latlon_bbox"])
            assert bounds is not None
            return RegionCoverage(
                event_id=str(raw["event_id"]).upper(),
                pixel_count=int(raw["pixel_count"]),
                pixel_box=tuple(int(item) for item in raw["pixel_bbox"]),
                bounds=bounds,
                center=_pixel_value(raw["region_centroid"]),
                areas=tuple(dict(item) for item in raw.get("administrative_coverage", [])),
                unresolved_pixels=int(raw.get("unresolved_pixel_count", 0)),
            )

        return self._invoke("geolocator.region", arguments, geolocate_pixel_region, convert)


class CasCast(BoundCapability):
    """CasCast ensemble grids bound to the event or upload in this request."""

    def forecast(
        self,
        event_id: str | None = None,
        horizon_hours: float = 1.0,
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
    ) -> Forecast:
        """Return a 1-, 2-, or 3-hour event forecast.

        Args: include_ground_truth should be true only for retrospective verification.
        Returns: Forecast with ``mean[lead,row,column]`` and ``members[member,lead,row,column]``.
        Example: ``fc = cascast.forecast(); spread = fc.members.std(axis=0)``
        """
        from ..backends.forecast import get_rolling_forecast_raw_values

        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
        arguments = {
            "event_id": self._event_id(event_id), "horizon_hours": horizon,
            "model_name": "cascast_live", "ensemble_members": int(ensemble_members),
            "include_ground_truth": bool(include_ground_truth), "array_transport": True,
        }
        return self._invoke("cascast.forecast", arguments, get_rolling_forecast_raw_values, _forecast_value)

    def forecast_upload(
        self,
        upload_id: str | None = None,
        horizon_hours: float = 1.0,
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
        force_live: bool = False,
    ) -> Forecast:
        """Return a forecast for the request-bound 13-frame upload.

        Args: force_live bypasses a matching model cache; normally leave it false.
        Returns: Forecast with directly computable NumPy arrays.
        Example: ``fc = cascast.forecast_upload(); peak = fc.mean.max(axis=(1, 2))``
        """
        from ..backends.forecast import get_uploaded_vil_sequence_forecast

        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0, or 3.0")
        arguments = {
            "upload_id": self._upload_id(upload_id), "model_name": "cascast_live",
            "ensemble_members": int(ensemble_members), "horizon_hours": horizon,
            "include_ground_truth": bool(include_ground_truth), "array_transport": True,
            "force_live": bool(force_live),
        }
        return self._invoke("cascast.forecast_upload", arguments, get_uploaded_vil_sequence_forecast, _forecast_value)


class Aurora(BoundCapability):
    """ERA5 analysis and Aurora regional fields bound to the current request."""

    def regional_fields(
        self,
        analysis_time_utc: str | None = None,
        forecast_leads_hours: tuple[int, ...] | None = None,
    ) -> EnvironmentFields:
        """Return aligned regional ERA5 and Aurora grids without hazard labels.

        Args: omitted time/leads use the values fixed when this object was constructed.
        Returns: EnvironmentFields with direct coordinate and field arrays by lead hour.
        Example: ``env = aurora.regional_fields(); t6 = env.forecast[6]['surface']['t2m_c']``
        """
        from ..backends.environment import regional_environment_fields

        time_value = str(analysis_time_utc or self.request.analysis_time_utc).strip()
        if not time_value:
            raise ValueError("this request has no bound analysis_time_utc")
        leads = tuple(int(item) for item in (forecast_leads_hours or self.request.forecast_leads_hours))
        arguments = {
            "event_id": self.request.event_id or None,
            "analysis_time_utc": time_value,
            "latitude": self.request.latitude,
            "longitude": self.request.longitude,
            "region": self.request.region,
            "forecast_leads": list(leads),
            "history_before_hours": self.request.history_before_hours,
            "history_after_hours": self.request.history_after_hours,
            "data_policy": self.request.data_policy,
        }

        def convert(raw: dict[str, Any]) -> EnvironmentFields:
            latitude = np.ascontiguousarray(raw["latitude"], dtype=np.float32)
            longitude = np.ascontiguousarray(raw["longitude"], dtype=np.float32)
            latitude.setflags(write=False)
            longitude.setflags(write=False)
            fields = {
                int(key): value for key, value in raw.get("aurora", {}).get("fields", {}).items()
            }
            lead_values = np.asarray(sorted(fields), dtype=np.int32)
            lead_values.setflags(write=False)
            return EnvironmentFields(
                event_id=str(raw.get("event_id") or "").upper() or None,
                analysis_time_utc=str(raw.get("analysis_time_utc") or time_value),
                latitude=latitude, longitude=longitude, era5=dict(raw.get("era5") or {}),
                forecast=fields, lead_hours=lead_values, cache_hit=bool(raw.get("cache_hit", False)),
            )

        return self._invoke(
            "aurora.regional_fields", arguments, regional_environment_fields, convert
        )
