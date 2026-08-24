from __future__ import annotations

"""Common request binding and call recording for public capabilities."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RequestBinding:
    event_id: str = ""
    upload_id: str = ""
    analysis_time_utc: str = ""
    latitude: float | None = None
    longitude: float | None = None
    region: dict[str, float] | None = None
    forecast_leads_hours: tuple[int, ...] = (6, 12, 18, 24)
    history_before_hours: int = 6
    history_after_hours: int = 24
    data_policy: str = "cached"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RequestBinding":
        raw = dict(value or {})
        leads = raw.get("forecast_leads_hours") or raw.get("forecast_leads") or (6, 12, 18, 24)
        region = raw.get("region")
        return cls(
            event_id=str(raw.get("event_id") or "").strip().upper(),
            upload_id=str(raw.get("sequence_upload_id") or raw.get("upload_id") or "").strip(),
            analysis_time_utc=str(raw.get("analysis_time_utc") or "").strip(),
            latitude=float(raw["latitude"]) if raw.get("latitude") is not None else None,
            longitude=float(raw["longitude"]) if raw.get("longitude") is not None else None,
            region={str(k): float(v) for k, v in region.items()} if isinstance(region, dict) else None,
            forecast_leads_hours=tuple(int(item) for item in leads),
            history_before_hours=int(raw.get("history_before_hours") or 6),
            history_after_hours=int(raw.get("history_after_hours") if raw.get("history_after_hours") is not None else 24),
            data_policy=str(raw.get("data_policy") or "cached").lower(),
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
        value = str(explicit or self.request.event_id).strip().upper()
        if not value:
            raise ValueError("this request has no bound SEVIR event")
        return value

    def _upload_id(self, explicit: str | None = None) -> str:
        value = str(explicit or self.request.upload_id).strip()
        if not value:
            raise ValueError("this request has no bound upload")
        return value

    def _invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
        convert: Callable[[dict[str, Any]], T],
    ) -> T:
        key = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
        if key in self._cache:
            value = self._cache[key]
            self._recorder(name, arguments, value, cache_hit=True)
            return value
        try:
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
            self._recorder(name, arguments, None, error)
            raise
        self._cache[key] = value
        self._recorder(name, arguments, value, cache_hit=bool(raw.get("cache_hit", False)))
        return value
