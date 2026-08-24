from __future__ import annotations

"""Four narrow capabilities exposed to generated Agent programs.

The classes in this module are deliberately thin data/model adapters.  They
do not choose thresholds, metrics, hazards, plots, or an analysis workflow.
Generated Python performs those question-specific decisions with NumPy and
matplotlib, as in Zephyrus.
"""

import json
from pathlib import Path
from typing import Any

from .base import WorldComponent


class _BoundCapability(WorldComponent):
    """Base for request-bound defaults; not part of the public Agent API."""

    def __init__(
        self,
        recorder,
        *,
        default_event_id: str = "",
        default_upload_id: str = "",
    ) -> None:
        super().__init__(recorder)
        self._default_event_id = str(default_event_id or "").strip().upper()
        self._default_upload_id = str(default_upload_id or "").strip()

    def _event_id(self, value: Any = None) -> str:
        candidate = value
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if isinstance(candidate, dict):
            candidate = candidate.get("event_id") or candidate.get("metadata", {}).get("event_id")
        resolved = str(candidate or self._default_event_id).strip().upper()
        if not resolved:
            raise ValueError("event_id is required because this request has no bound SEVIR event")
        return resolved

    def _upload_id(self, value: Any = None) -> str:
        candidate = value
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if isinstance(candidate, dict):
            candidate = candidate.get("upload_id") or candidate.get("metadata", {}).get("upload_id")
        resolved = str(candidate or self._default_upload_id).strip()
        if not resolved:
            raise ValueError("upload_id is required because this request has no bound upload")
        return resolved


class SevirCapability(_BoundCapability):
    """Read SEVIR observations, catalog metadata, and historical candidates.

    This object only retrieves evidence. It does not compute weather labels,
    thresholds, rankings, uncertainty, verification scores, or plots. Methods
    whose ``event_id`` is omitted use the event already bound to this request.
    Uploaded pixel arrays are already available in ``ds``; ``match_upload`` is
    only for resolving an upload to a catalog event.
    """

    def event(self, event_id: str | None = None) -> dict[str, Any]:
        """Return catalog metadata and the 49-frame UTC time axis for one event."""
        from ..data_access.catalog import get_sevir_event_metadata

        resolved = self._event_id(event_id)
        return self._call("sevir.event", {"event_id": resolved}, get_sevir_event_metadata)

    def search(
        self,
        query: str = "",
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search SEVIR catalog metadata; this method does not load pixel grids."""
        from ..data_access.catalog import search_sevir_catalog

        arguments = {
            "query": query, "year": year, "month": month,
            "event_type": event_type, "limit": limit,
        }
        return self._call("sevir.search", arguments, search_sevir_catalog)

    def nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100.0,
        year: int | None = None,
        month: int | None = None,
        event_type: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find catalog events inside a WGS84 radius; returns metadata only."""
        from ..data_access.catalog import search_sevir_catalog_by_region

        arguments = {
            "latitude": float(latitude), "longitude": float(longitude),
            "radius_km": float(radius_km), "year": year, "month": month,
            "event_type": event_type, "limit": limit,
        }
        return self._call("sevir.nearby", arguments, search_sevir_catalog_by_region)

    def observations(
        self,
        event_id: str | None = None,
        start_frame: int = 0,
        end_frame: int = 49,
    ) -> dict[str, Any]:
        """Load encoded-VIL ``[time,row,column]`` frames and relative minutes.

        Pixel value 255 is missing and must be excluded from calculations.
        ``end_frame`` follows the existing SEVIR loader contract. Large arrays
        remain inside the Worker and should be reduced before assigning result.
        """
        from ..data_access.catalog import load_sevir_vil_event

        resolved = self._event_id(event_id)
        arguments = {
            "event_id": resolved,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
        }
        return self._call("sevir.observations", arguments, load_sevir_vil_event)

    def frame_histograms(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
    ) -> dict[str, Any]:
        """Load exact 0..255 per-frame histograms for up to ten events."""
        from ..data_access.catalog import load_sevir_vil_frame_histograms

        arguments = {
            "event_ids": [str(item).strip().upper() for item in event_ids],
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
        }
        return self._call("sevir.frame_histograms", arguments, load_sevir_vil_frame_histograms)

    def similar_events(
        self,
        event_id: str | None = None,
        field: str = "data_max",
        limit: int = 10,
        same_event_type: bool = True,
        local_hdf5_only: bool = True,
    ) -> dict[str, Any]:
        """Return metadata candidates ranked against the bound/reference event.

        This is candidate retrieval, not a final meteorological similarity
        judgment. Generated code decides which fields and comparisons matter.
        """
        from ..backends.statistics import find_similar_event_candidates

        resolved = self._event_id(event_id)
        arguments = {
            "reference_event_id": resolved,
            "field": field,
            "candidate_limit": int(limit),
            "same_event_type": bool(same_event_type),
            "local_hdf5_only": bool(local_hdf5_only),
        }
        return self._call("sevir.similar_events", arguments, find_similar_event_candidates)

    def historical_observations(
        self,
        event_ids: list[str],
        start_frame: int = 0,
        end_frame: int = 49,
        spatial_stride: int = 1,
        max_total_values: int = 8_000_000,
    ) -> dict[str, Any]:
        """Load bounded raw grids for Agent-selected historical comparisons."""
        from ..backends.statistics import load_event_history_raw

        arguments = {
            "event_ids": [str(item).strip().upper() for item in event_ids],
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "spatial_stride": int(spatial_stride),
            "max_total_values": int(max_total_values),
        }
        return self._call("sevir.historical_observations", arguments, load_event_history_raw)

    def match_upload(self, upload_id: str | None = None) -> dict[str, Any]:
        """Resolve a bound upload to an exact/verified SEVIR catalog event."""
        from ..data_access.uploads import match_uploaded_event

        resolved = self._upload_id(upload_id)
        return self._call("sevir.match_upload", {"upload_id": resolved}, match_uploaded_event)


class GeolocatorCapability(_BoundCapability):
    """Convert between SEVIR pixels, WGS84 coordinates, places, and US areas.

    This object performs coordinate lookup only. It does not detect storms or
    decide which region is meteorologically important. Omitted ``event_id``
    arguments use the event already bound to the request.
    """

    def event(
        self,
        event_id: str | None = None,
        row: int | None = None,
        column: int | None = None,
    ) -> dict[str, Any]:
        """Return event bounds/center and optionally one projected pixel point."""
        from ..backends.geospatial import geolocate_sevir_event

        resolved = self._event_id(event_id)
        arguments = {"event_id": resolved, "row": row, "column": column}
        return self._call("geolocator.event", arguments, geolocate_sevir_event)

    def pixel(
        self,
        row: int,
        column: int,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Return WGS84 and projected coordinates for one zero-based pixel."""
        location = self.event(event_id=event_id, row=int(row), column=int(column))
        return location["pixel_location"]

    def place(self, name: str, country_codes: str = "us") -> dict[str, Any]:
        """Resolve a place name or ``latitude,longitude`` text to WGS84."""
        from ..backends.geospatial import resolve_place

        arguments = {"place": str(name), "country_codes": str(country_codes)}
        return self._call("geolocator.place", arguments, resolve_place)

    def administrative_area(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Resolve a WGS84 point to bundled US state and county polygons."""
        from ..backends.geospatial import reverse_geocode_us

        arguments = {"latitude": float(latitude), "longitude": float(longitude)}
        return self._call("geolocator.administrative_area", arguments, reverse_geocode_us)

    def region(
        self,
        pixels: list[list[int]],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve state/county coverage for Agent-selected SEVIR pixels."""
        from ..backends.geospatial import geolocate_pixel_region

        resolved = self._event_id(event_id)
        arguments = {"event_id": resolved, "pixels": pixels}
        return self._call("geolocator.region", arguments, geolocate_pixel_region)


class CasCastCapability(_BoundCapability):
    """Return CasCast ensemble predictions for events or uploaded sequences.

    The object returns raw forecast/member grids and aligned lead times. It
    deliberately computes no threshold categories, uncertainty summaries,
    verification scores, tracks, hazard labels, or plots; generated Python
    performs only the calculations required by the user's question.
    """

    def forecast(
        self,
        event_id: str | None = None,
        horizon_hours: float = 1.0,
        model_name: str = "cascast",
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
    ) -> dict[str, Any]:
        """Return event forecast mean/members for a 1, 2, or 3 hour horizon.

        ``include_ground_truth`` must remain false unless the user explicitly
        asks for retrospective verification. Cached R190 data is reused when
        its identity and model parameters match.
        """
        from ..backends.forecast import get_rolling_forecast_raw_values

        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0 or 3.0")
        resolved = self._event_id(event_id)
        arguments = {
            "event_id": resolved,
            "model_name": str(model_name),
            "horizon_hours": horizon,
            "ensemble_members": int(ensemble_members),
            "include_ground_truth": bool(include_ground_truth),
            "array_transport": True,
        }
        return self._call("cascast.forecast", arguments, get_rolling_forecast_raw_values)

    def forecast_upload(
        self,
        upload_id: str | None = None,
        horizon_hours: float = 1.0,
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
        force_live: bool = False,
    ) -> dict[str, Any]:
        """Run/read CasCast for a bound 13-frame upload and return raw grids.

        This method is invalid for a single-frame upload. Ground truth is
        excluded unless retrospective verification was explicitly requested.
        """
        from ..backends.forecast import get_uploaded_vil_sequence_forecast

        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0 or 3.0")
        resolved = self._upload_id(upload_id)
        arguments = {
            "upload_id": resolved,
            "model_name": "cascast_live",
            "ensemble_members": int(ensemble_members),
            "horizon_hours": horizon,
            "include_ground_truth": bool(include_ground_truth),
            "force_live": bool(force_live),
            "array_transport": True,
        }
        return self._call("cascast.forecast_upload", arguments, get_uploaded_vil_sequence_forecast)


class AuroraCapability(_BoundCapability):
    """Read cached or live ERA5/Aurora large-scale environmental evidence.

    Aurora covers regional environmental evolution at 6-24 hour lead times.
    It does not retrieve SEVIR pixels, run CasCast, label hazards, select a
    local target, or compare the two models. Generated Python screens the raw
    regional fields and then explicitly calls the other capabilities if needed.
    """

    def case(self, event_id: str | None = None) -> dict[str, Any]:
        """Return registered ERA5/Aurora evidence for a cached study event."""
        from ..data_access.config import ROOT
        from ..data_access.foundation import get_case_foundation_artifacts
        from ..data_access.foundation_cache import foundation_cache_status
        from ..data_access.foundation_validation import build_foundation_bundle

        resolved = self._event_id(event_id)

        def load_case(event_id: str) -> dict[str, Any]:
            registry = get_case_foundation_artifacts(event_id)
            evidence_path = Path(ROOT) / "data" / "era5_cases" / event_id / "artifacts" / "case_evidence.json"
            evidence = (
                json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence_path.is_file()
                else registry.get("evidence")
            )
            bundle = build_foundation_bundle(event_id, evidence, registry.get("artifacts", []))
            return {
                "status": bundle["status"] if registry.get("status") == "ok" else registry.get("status", "not_found"),
                "event_id": event_id,
                "evidence": evidence,
                "evidence_bundle": bundle,
                "cache": foundation_cache_status(event_id),
                "validation": bundle["validation"],
                "artifacts": registry.get("artifacts", []),
            }

        return self._call("aurora.case", {"event_id": resolved}, load_case)

    def forecast(
        self,
        analysis_time_utc: str,
        event_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        region: dict[str, float] | None = None,
        forecast_leads_hours: tuple[int, ...] = (6, 12, 18, 24),
        history_before_hours: int = 6,
        history_after_hours: int = 24,
    ) -> dict[str, Any]:
        """Fetch ERA5 inputs when needed and run Aurora for requested leads."""
        from .live_foundation import forecast_aurora

        arguments = {
            "event_id": self._event_id(event_id) if (event_id or self._default_event_id) else None,
            "analysis_time_utc": str(analysis_time_utc),
            "latitude": latitude,
            "longitude": longitude,
            "region": region,
            "forecast_leads": [int(item) for item in forecast_leads_hours],
            "history_before_hours": int(history_before_hours),
            "history_after_hours": int(history_after_hours),
        }
        return self._call("aurora.forecast", arguments, forecast_aurora)

    def regional_fields(
        self,
        analysis_time_utc: str,
        event_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        region: dict[str, float] | None = None,
        forecast_leads_hours: tuple[int, ...] = (6, 12, 18, 24),
        history_before_hours: int = 6,
        history_after_hours: int = 24,
    ) -> dict[str, Any]:
        """Return aligned ERA5/Aurora regional grids without hazard screening.

        The result contains latitude/longitude coordinates, ERA5 fields and
        Aurora fields by lead hour. Generated code chooses variables, scores,
        thresholds, candidate regions, and any subsequent SEVIR lookup.
        """
        from .live_foundation import region_fields

        arguments = {
            "event_id": self._event_id(event_id) if (event_id or self._default_event_id) else None,
            "analysis_time_utc": str(analysis_time_utc),
            "latitude": latitude,
            "longitude": longitude,
            "region": region,
            "forecast_leads": [int(item) for item in forecast_leads_hours],
            "history_before_hours": int(history_before_hours),
            "history_after_hours": int(history_after_hours),
        }
        return self._call("aurora.regional_fields", arguments, region_fields)
