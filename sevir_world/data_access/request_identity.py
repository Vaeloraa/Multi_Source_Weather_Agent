"""Canonical request identities for model/data cache reuse."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_request_payload(
    *, event_id: str = "", analysis_time_utc: str = "", latitude: Any = None,
    longitude: Any = None, region: dict[str, Any] | None = None,
    variables: list[str] | tuple[str, ...] | None = None,
    pressure_levels: list[int] | tuple[int, ...] | None = None,
    history_before_hours: Any = None, history_after_hours: Any = None,
    forecast_leads_hours: list[int] | tuple[int, ...] | None = None,
    model: str = "", model_version: str = "", resolution: str = "",
    task: str = "", evidence_kind: str = "",
) -> dict[str, Any]:
    return _clean({
        "event_id": str(event_id or "").strip().upper(),
        "analysis_time_utc": str(analysis_time_utc or "").strip(),
        "latitude": latitude, "longitude": longitude,
        "region": region or {},
        "variables": sorted({str(x) for x in (variables or [])}),
        "pressure_levels": sorted({int(x) for x in (pressure_levels or [])}),
        "history_before_hours": history_before_hours,
        "history_after_hours": history_after_hours,
        "forecast_leads_hours": sorted({int(x) for x in (forecast_leads_hours or [])}),
        "model": str(model or ""), "model_version": str(model_version or ""),
        "resolution": str(resolution or ""), "task": str(task or ""),
        "evidence_kind": str(evidence_kind or ""),
    })


def request_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def request_identity(**kwargs: Any) -> dict[str, Any]:
    canonical = canonical_request_payload(**kwargs)
    return {"request_key": request_key(canonical), "canonical": canonical}


def request_compatible(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compatibility means the requested evidence fields are identical."""
    return request_key(expected) == request_key(actual)


def detect_event_conflict(question: str, payload: dict[str, Any], event_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect explicit time/location input that conflicts with upload metadata."""
    text = str(question or "")
    upload_event = str(payload.get("uploaded_event_id") or payload.get("event_id") or "").strip().upper()
    explicit_ids = [value.upper() for value in re.findall(r"\bR\d{14}\b", text, flags=re.IGNORECASE)]
    record_time = str((event_record or {}).get("time_utc") or "")
    explicit_dates = re.findall(r"\b((?:19|20)\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    explicit_years = {int(year) for year, _, _ in explicit_dates}
    record_year_match = re.match(r"(\d{4})", record_time)
    record_year = int(record_year_match.group(1)) if record_year_match else None
    reasons: list[str] = []
    if explicit_ids and upload_event and any(value != upload_event for value in explicit_ids):
        reasons.append("explicit_event_id_conflict")
    if explicit_years and record_year and record_year not in explicit_years:
        reasons.append("explicit_year_conflict")
    if explicit_dates and record_time:
        record_date = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", record_time)
        if record_date:
            record_tuple = tuple(int(value) for value in record_date.groups())
            explicit_tuples = {tuple(int(value) for value in item) for item in explicit_dates}
            if record_tuple not in explicit_tuples:
                reasons.append("explicit_date_conflict")
    return {
        "status": "conflict" if reasons else "compatible",
        "upload_event_id": upload_event,
        "explicit_event_ids": explicit_ids,
        "explicit_years": sorted(explicit_years),
        "reasons": reasons,
    }
