"""Normalize explicit request context without a demo-place lookup table."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def _iso_utc(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace(" UTC", "Z").replace(" utc", "Z")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_analysis_time(question: str) -> str | None:
    text = str(question or "")
    iso = re.search(
        r"(?<!\d)(20\d{2}-\d{1,2}-\d{1,2})[T\s]+"
        r"(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*(?:UTC|Z))?",
        text,
        flags=re.IGNORECASE,
    )
    if iso:
        return _iso_utc(f"{iso.group(1)}T{iso.group(2)}Z")
    chinese = re.search(
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日(?:\s*|，|,)*"
        r"(\d{1,2})(?::|时)(\d{1,2})?(?:分)?(?:\s*UTC)?",
        text,
        flags=re.IGNORECASE,
    )
    if not chinese:
        return None
    return _iso_utc(
        f"{int(chinese.group(1)):04d}-{int(chinese.group(2)):02d}-"
        f"{int(chinese.group(3)):02d}T{int(chinese.group(4)):02d}:"
        f"{int(chinese.group(5) or 0):02d}:00Z"
    )


def resolve_known_place(value: str) -> dict[str, Any] | None:
    """Compatibility hook: all place names now use the configured geocoder."""

    del value
    return None


def enrich_request_context(question: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not str(payload.get("analysis_time_utc") or "").strip():
        extracted = extract_analysis_time(question)
        if extracted:
            payload["analysis_time_utc"] = extracted
            payload["analysis_time_source"] = "question"
    else:
        normalized = _iso_utc(str(payload.get("analysis_time_utc")))
        if normalized:
            payload["analysis_time_utc"] = normalized

    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if latitude is not None and longitude is not None:
        return {
            "status": "ok",
            "source": "explicit_coordinates",
            "analysis_time_utc": payload.get("analysis_time_utc"),
            "latitude": latitude,
            "longitude": longitude,
        }
    return {
        "status": "unresolved",
        "place_name": str(payload.get("place_name") or "").strip(),
        "analysis_time_utc": payload.get("analysis_time_utc"),
    }


def known_place_candidates() -> list[dict[str, Any]]:
    return []
