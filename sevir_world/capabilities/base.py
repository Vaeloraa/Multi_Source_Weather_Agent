from __future__ import annotations

"""Common request binding and call recording for public capabilities."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def _utc_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace(" UTC", "Z").replace(" utc", "Z")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RegionBounds:
    """Validated WGS84 region in south-west-north-east order."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("region must satisfy -90 <= south < north <= 90")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("region must satisfy -180 <= west < east <= 180")

    @classmethod
    def from_value(cls, value: Any) -> "RegionBounds | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("region must be an object with south, west, north and east")
        required = {"south", "west", "north", "east"}
        if set(value) != required:
            raise ValueError("region must contain exactly south, west, north and east")
        return cls(**{key: float(value[key]) for key in required})

    def as_dict(self) -> dict[str, float]:
        return {
            "south": self.south,
            "west": self.west,
            "north": self.north,
            "east": self.east,
        }


@dataclass(frozen=True, slots=True)
class RequestAssets:
    """Asset identities selected for one request."""

    event_id: str = ""
    upload_id: str = ""
    sevir_subject_key: str = ""


@dataclass(frozen=True, slots=True)
class RequestTarget:
    """User-requested time and geographic target."""

    requested_time_utc: str = ""
    latitude: float | None = None
    longitude: float | None = None
    region: RegionBounds | None = None


@dataclass(frozen=True, slots=True)
class ForecastSpecification:
    """Aurora initialization, regional domain, leads and history windows."""

    initialization_time_utc: str = ""
    region: RegionBounds | None = None
    lead_hours: tuple[int, ...] = (6, 12, 18, 24)
    history_before_hours: int = 6
    history_after_hours: int = 24


@dataclass(frozen=True, slots=True)
class RequestProvenance:
    """Ingress source of each resolved request-fact group."""

    event: str = ""
    upload: str = ""
    requested_time: str = ""
    location: str = ""
    forecast: str = ""


@dataclass(frozen=True, slots=True)
class DataAccessPolicy:
    """Whether capability backends may acquire live data or use cache only."""

    mode: str = "cached"


@dataclass(frozen=True, slots=True)
class RequestBinding:
    """Immutable request facts shared by all request-bound capabilities.

    Public capability methods use these values whenever their ordinary Python
    signatures omit an event, upload, location, time, forecast window or data
    policy. The same binding is exposed as each capability's ``request``
    attribute and remains fixed for the complete execution session.
    """

    event_id: str = ""
    upload_id: str = ""
    sevir_subject_key: str = ""
    requested_time_utc: str = ""
    aurora_initialization_time_utc: str = ""
    aurora_region: RegionBounds | None = None
    latitude: float | None = None
    longitude: float | None = None
    region: RegionBounds | None = None
    forecast_leads_hours: tuple[int, ...] = (6, 12, 18, 24)
    history_before_hours: int = 6
    history_after_hours: int = 24
    data_policy: str = "cached"
    provenance: RequestProvenance = field(default_factory=RequestProvenance)

    def __post_init__(self) -> None:
        sevir_subject_key = str(self.sevir_subject_key or "").strip().lower()
        event_id = str(self.event_id or "").strip().upper()
        upload_id = str(self.upload_id or "").strip()
        requested_time = _utc_text(self.requested_time_utc, "requested_time_utc")
        initialization = _utc_text(
            self.aurora_initialization_time_utc,
            "aurora_initialization_time_utc",
        )
        aurora_region = RegionBounds.from_value(self.aurora_region)
        latitude = None if self.latitude is None else float(self.latitude)
        longitude = None if self.longitude is None else float(self.longitude)
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        if latitude is not None and not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if longitude is not None and not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")
        region = RegionBounds.from_value(self.region)
        leads = tuple(int(item) for item in self.forecast_leads_hours)
        if not leads or any(item <= 0 for item in leads):
            raise ValueError("forecast_leads_hours must contain positive hours")
        before = int(self.history_before_hours)
        after = int(self.history_after_hours)
        if before < 0 or after < 0:
            raise ValueError("history windows must be non-negative hours")
        policy = str(self.data_policy or "cached").strip().lower()
        if policy not in {"cached", "live"}:
            raise ValueError("data_policy must be cached or live")
        provenance = self.provenance
        if isinstance(provenance, dict):
            provenance = RequestProvenance(**{
                key: str(provenance.get(key) or "")
                for key in ("event", "upload", "requested_time", "location", "forecast")
            })
        if not isinstance(provenance, RequestProvenance):
            raise ValueError("provenance must be RequestProvenance or an object")
        for name, value in (
            ("sevir_subject_key", sevir_subject_key),
            ("event_id", event_id),
            ("upload_id", upload_id),
            ("requested_time_utc", requested_time),
            ("aurora_initialization_time_utc", initialization),
            ("aurora_region", aurora_region),
            ("latitude", latitude),
            ("longitude", longitude),
            ("region", region),
            ("forecast_leads_hours", leads),
            ("history_before_hours", before),
            ("history_after_hours", after),
            ("data_policy", policy),
            ("provenance", provenance),
        ):
            object.__setattr__(self, name, value)

    @property
    def assets(self) -> RequestAssets:
        """Return the request's selected event and upload identities."""

        return RequestAssets(
            event_id=self.event_id,
            upload_id=self.upload_id,
            sevir_subject_key=self.sevir_subject_key,
        )

    @property
    def target(self) -> RequestTarget:
        """Return the user-requested time and geographic target."""

        return RequestTarget(
            self.requested_time_utc,
            self.latitude,
            self.longitude,
            self.region,
        )

    @property
    def forecast(self) -> ForecastSpecification:
        """Return forecast initialization, leads and history windows."""

        return ForecastSpecification(
            self.aurora_initialization_time_utc,
            self.aurora_region,
            self.forecast_leads_hours,
            self.history_before_hours,
            self.history_after_hours,
        )

    @property
    def policy(self) -> DataAccessPolicy:
        """Return the request's data-access policy."""

        return DataAccessPolicy(self.data_policy)

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RequestBinding":
        raw = dict(value or {})
        upload_id = str(raw.get("upload_id") or "").strip()
        sequence_upload_id = str(raw.get("sequence_upload_id") or "").strip()
        if upload_id and sequence_upload_id and upload_id != sequence_upload_id:
            raise ValueError("upload_id conflicts with sequence_upload_id")
        leads_hours = raw.get("forecast_leads_hours")
        legacy_leads = raw.get("forecast_leads")
        if leads_hours is not None and legacy_leads is not None:
            if tuple(int(item) for item in leads_hours) != tuple(int(item) for item in legacy_leads):
                raise ValueError("forecast_leads_hours conflicts with forecast_leads")
        leads = (
            leads_hours if leads_hours is not None
            else legacy_leads if legacy_leads is not None
            else (6, 12, 18, 24)
        )
        event_resolution = raw.get("event_resolution")
        place_resolution = raw.get("place_resolution")
        provenance = raw.get("provenance")
        if not isinstance(provenance, (dict, RequestProvenance)):
            provenance = RequestProvenance(
                event=str(event_resolution.get("source") or "")
                if isinstance(event_resolution, dict) else "",
                upload="request" if upload_id or sequence_upload_id else "",
                requested_time=str(raw.get("requested_time_source") or ""),
                location=(
                    str(raw.get("location_source") or "")
                    or (
                        str(place_resolution.get("source") or "")
                        if isinstance(place_resolution, dict) else ""
                    )
                ),
                forecast="request" if (
                    raw.get("aurora_initialization_time_utc")
                    or leads_hours is not None
                    or legacy_leads is not None
                ) else "default",
            )
        return cls(
            sevir_subject_key=str(raw.get("sevir_subject_key") or "").strip().lower(),
            event_id=str(raw.get("event_id") or "").strip().upper(),
            upload_id=sequence_upload_id or upload_id,
            requested_time_utc=str(raw.get("requested_time_utc") or "").strip(),
            aurora_initialization_time_utc=str(
                raw.get("aurora_initialization_time_utc") or ""
            ).strip(),
            aurora_region=RegionBounds.from_value(
                raw.get("aurora_region")
                if raw.get("aurora_region") is not None
                else raw.get("region")
            ),
            latitude=float(raw["latitude"]) if raw.get("latitude") is not None else None,
            longitude=float(raw["longitude"]) if raw.get("longitude") is not None else None,
            region=RegionBounds.from_value(raw.get("region")),
            forecast_leads_hours=tuple(int(item) for item in leads),
            history_before_hours=int(
                raw.get("history_before_hours")
                if raw.get("history_before_hours") is not None else 6
            ),
            history_after_hours=int(raw.get("history_after_hours") if raw.get("history_after_hours") is not None else 24),
            data_policy=str(raw.get("data_policy") or "cached").lower(),
            provenance=provenance,
        )


class CapabilityUnavailable(RuntimeError):
    pass


class BoundCapability:
    """One capability whose request identity is fixed at construction."""

    def __init__(self, recorder: Callable[..., None], binding: RequestBinding) -> None:
        self._recorder = recorder
        self.request = binding
        self._cache: dict[str, Any] = {}

    def _event_id(self, explicit: str | None = None) -> str:
        candidate: Any = explicit or self.request.assets.event_id
        # Public catalog searches return CatalogEvent values. Accepting their
        # documented ``id`` attribute at the boundary prevents generated code
        # from accidentally stringifying the entire dataclass as an event ID.
        if not isinstance(candidate, (str, int, float)):
            candidate = getattr(candidate, "id", None)
            if candidate is None and isinstance(explicit, dict):
                candidate = explicit.get("id") or explicit.get("event_id")
        value = str(candidate or "").strip().upper()
        if not value:
            raise ValueError("this request has no bound SEVIR event")
        return value

    def _upload_id(self, explicit: str | None = None) -> str:
        value = str(explicit or self.request.assets.upload_id).strip()
        if not value:
            raise ValueError("this request has no bound upload")
        return value

    def _invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
        convert: Callable[[dict[str, Any]], T],
        *,
        public_arguments: dict[str, Any] | None = None,
    ) -> T:
        recorded_arguments = arguments if public_arguments is None else public_arguments
        key = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
        try:
            cache_hit = key in self._cache
            reserve = getattr(self._recorder, "reserve", None)
            cost_units = reserve(name, recorded_arguments, cache_hit=cache_hit) if callable(reserve) else None
            if cache_hit:
                value = self._cache[key]
                self._recorder(
                    name, recorded_arguments, value, cache_hit=True, cost_units=cost_units
                )
                return value
            raw = function(**arguments)
            if not isinstance(raw, dict):
                raise TypeError(f"{name} backend returned {type(raw).__name__}, expected dict")
            status = str(raw.get("status", "ok")).lower()
            if status not in {
                "ok", "ambiguous", "approximate_boundary_match", "partial",
                "outside_us_boundaries", "boundary_data_unavailable",
            }:
                reason = raw.get("reason") or raw.get("error") or status
                raise CapabilityUnavailable(f"{name} unavailable: {reason}")
            value = convert(raw)
        except Exception as error:
            self._recorder(
                name, recorded_arguments, None, error,
                cost_units=locals().get("cost_units"),
            )
            raise
        self._cache[key] = value
        self._recorder(
            name, recorded_arguments, value,
            cache_hit=bool(raw.get("cache_hit", False)),
            cost_units=cost_units,
        )
        return value
    def _subject_key(self, explicit: str | None = None) -> str:
        """Return the unique request-bound scientific subject identity."""

        candidate: Any = explicit or self.request.assets.sevir_subject_key
        if not candidate and self.request.assets.event_id:
            from ..tools.sevir.Catalog import resolve_sevir_subject_key

            candidate = resolve_sevir_subject_key(self.request.assets.event_id)
        if not isinstance(candidate, (str, int, float)):
            candidate = getattr(candidate, "subject_key", None)
            if candidate is None and isinstance(explicit, dict):
                candidate = explicit.get("subject_key")
        value = str(candidate or "").strip().lower()
        if not value:
            raise ValueError("this request has no uniquely bound scientific subject")
        return value
