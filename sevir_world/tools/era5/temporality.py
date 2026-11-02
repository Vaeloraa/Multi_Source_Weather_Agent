from __future__ import annotations

"""ERA5 product temporality and availability boundaries."""

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


ERA5T_MINIMUM_LAG_DAYS = 5
ERA5_FINAL_CONSERVATIVE_LAG_DAYS = 90


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        if not text:
            raise ValueError("ERA5 valid time is required")
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_era5_valid_times(
    valid_times: Iterable[datetime | str],
    *,
    as_of_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """Classify requested ERA5 times without calling a remote service."""

    values = [_utc(value) for value in valid_times]
    if not values:
        raise ValueError("at least one ERA5 valid time is required")
    reference = _utc(as_of_utc) if as_of_utc is not None else datetime.now(timezone.utc)
    earliest = min(values)
    latest = max(values)
    age = reference - latest
    age_hours = age.total_seconds() / 3600.0

    common = {
        "is_real_time": False,
        "earliest_valid_time_utc": earliest.isoformat().replace("+00:00", "Z"),
        "latest_valid_time_utc": latest.isoformat().replace("+00:00", "Z"),
        "as_of_utc": reference.isoformat().replace("+00:00", "Z"),
        "age_hours_at_classification": round(age_hours, 3),
    }
    if age < timedelta(days=ERA5T_MINIMUM_LAG_DAYS):
        return {
            **common,
            "product": "ERA5_OR_ERA5T_NOT_YET_AVAILABLE",
            "data_class": "unavailable_recent_reanalysis",
            "availability_status": "not_yet_available",
            "provisional": None,
            "minimum_lag_days": ERA5T_MINIMUM_LAG_DAYS,
            "label_zh": "ERA5/ERA5T is not yet available under the release schedule",
        }
    if age < timedelta(days=ERA5_FINAL_CONSERVATIVE_LAG_DAYS):
        return {
            **common,
            "product": "ERA5T",
            "data_class": "preliminary_reanalysis",
            "availability_status": "available_with_revision_risk",
            "provisional": True,
            "minimum_lag_days": ERA5T_MINIMUM_LAG_DAYS,
            "label_zh": "ERA5T initial reanalysis, approximately five days delayed and subject to revision",
        }
    return {
        **common,
        "product": "ERA5",
        "data_class": "final_reanalysis",
        "availability_status": "available",
        "provisional": False,
        "minimum_lag_days": ERA5_FINAL_CONSERVATIVE_LAG_DAYS,
        "label_zh": "ERA5 final historical reanalysis",
    }


def classify_era5_request(
    analysis_time_utc: datetime | str,
    *,
    history_before_hours: int = 0,
    history_after_hours: int = 0,
    forecast_leads_hours: Iterable[int] = (),
    as_of_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """Classify the newest valid time needed by one ERA5 request."""

    analysis = _utc(analysis_time_utc)
    offsets = [0, max(0, int(history_after_hours))]
    offsets.extend(max(0, int(value)) for value in forecast_leads_hours)
    newest = analysis + timedelta(hours=max(offsets))
    earliest = analysis - timedelta(hours=max(0, int(history_before_hours)))
    result = classify_era5_valid_times((earliest, newest), as_of_utc=as_of_utc)
    result["analysis_time_utc"] = analysis.isoformat().replace("+00:00", "Z")
    result["maximum_requested_offset_hours"] = max(offsets)
    return result
