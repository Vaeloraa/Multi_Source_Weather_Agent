"""Cache identity for the physical ERA5/Aurora request bundle."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def request_identity(
    *, analysis_time_utc: str = "", latitude: Any = None,
    longitude: Any = None, region: dict[str, Any] | None = None,
    variables: list[str] | tuple[str, ...] | None = None,
    pressure_levels: list[int] | tuple[int, ...] | None = None,
    history_before_hours: Any = None, history_after_hours: Any = None,
    forecast_leads_hours: list[int] | tuple[int, ...] | None = None,
    model: str = "", model_version: str = "", resolution: str = "",
    task: str = "", evidence_kind: str = "",
) -> dict[str, Any]:
    """Preserve existing bundle keys while ignoring study/event labels."""
    canonical = {
        "event_id": "",
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
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "request_key": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32],
        "canonical": canonical,
    }
