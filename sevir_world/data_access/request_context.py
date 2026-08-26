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


def extract_requested_time(question: str) -> str | None:
    """Extract the time explicitly requested by the user, without model semantics."""

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
    # Request facts are established once at ingress.  Catalog/cache evidence
    # must never be allowed to overwrite them later in the pipeline.
    supplied_raw = str(payload.get("requested_time_utc") or "").strip()
    supplied = _iso_utc(supplied_raw)
    if supplied_raw and supplied is None:
        return {
            "status": "invalid",
            "field": "requested_time_utc",
            "value": supplied_raw,
        }
    extracted = extract_requested_time(question)
    if supplied and extracted and supplied != extracted:
        payload["context_conflict"] = {
            "field": "requested_time_utc",
            "payload": supplied,
            "question": extracted,
        }
        return {"status": "conflict", "field": "requested_time_utc", "payload": supplied, "question": extracted}
    if supplied:
        payload["requested_time_utc"] = supplied
        payload["requested_time_source"] = payload.get("requested_time_source", "request")
    elif extracted:
        payload["requested_time_utc"] = extracted
        payload["requested_time_source"] = "question"
    # ``requested_time_utc`` is the user's question time. Aurora initialization
    # is a separate forecast identity and must be supplied/resolved explicitly.
    initialization_raw = str(payload.get("aurora_initialization_time_utc") or "").strip()
    if initialization_raw:
        initialization = _iso_utc(initialization_raw)
        if initialization is None:
            return {
                "status": "invalid",
                "field": "aurora_initialization_time_utc",
                "value": initialization_raw,
            }
        payload["aurora_initialization_time_utc"] = initialization
    elif payload.get("requested_time_utc") and re.search(
        # Accept the common ``auraro`` typo used in the demo UI/chat. The
        # capability itself is still Aurora; this only affects ingress-time
        # binding of the requested analysis time.
        r"Aurora|auraro", str(question or ""), flags=re.IGNORECASE
    ):
        # In a self-contained forecast request such as “现在是 X，请用 Aurora
        # 判断未来…”, X is the forecast initialization identity. The web UI
        # exposes one analysis-time field, so requiring a second hidden field
        # would make every fresh Aurora request impossible while old study
        # cases appeared to work only because they leaked an initialization.
        payload["aurora_initialization_time_utc"] = payload["requested_time_utc"]
        payload["aurora_initialization_time_source"] = "requested_time_for_aurora"

    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    if (latitude is None) != (longitude is None):
        return {
            "status": "invalid",
            "field": "latitude_longitude",
            "value": {"latitude": latitude, "longitude": longitude},
        }
    if latitude is not None and longitude is not None:
        payload["location_source"] = "explicit_coordinates"
        return {
            "status": "ok",
            "facts": "request",
            "source": "explicit_coordinates",
            "requested_time_utc": payload.get("requested_time_utc"),
            "aurora_initialization_time_utc": payload.get("aurora_initialization_time_utc"),
            "latitude": latitude,
            "longitude": longitude,
        }
    return {
        "status": "unresolved",
        "facts": "request",
        "place_name": str(payload.get("place_name") or "").strip(),
        "requested_time_utc": payload.get("requested_time_utc"),
        "aurora_initialization_time_utc": payload.get("aurora_initialization_time_utc"),
    }


def known_place_candidates() -> list[dict[str, Any]]:
    return []
