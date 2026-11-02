from __future__ import annotations

"""Forward geocoding with explicit provenance and no dataset dependency."""

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from .boundaries import point_to_location


DEFAULT_GEOCODER_URL = "https://nominatim.openstreetmap.org/search"


def resolve_place(
    place: str, *, country_codes: str = "", geocoder_url: str | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    text = str(place or "").strip()
    if not text:
        raise ValueError("place must not be empty")
    tokens = [item.strip() for item in text.replace(";", ",").split(",")]
    if len(tokens) == 2:
        try:
            latitude, longitude = float(tokens[0]), float(tokens[1])
        except ValueError:
            latitude = longitude = None
        if latitude is not None and longitude is not None:
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError("explicit coordinates are outside WGS84")
            location = point_to_location(latitude, longitude)
            return {
                "status": "ok", "query": text, "latitude": latitude,
                "longitude": longitude, "display_name": text,
                "location": location, "candidates": [],
                "provenance": {"source": "explicit_wgs84_coordinates"},
            }
    endpoint = str(geocoder_url or os.getenv("GEOSPATIAL_GEOCODER_URL", DEFAULT_GEOCODER_URL)).strip()
    parameters = {"q": text, "format": "jsonv2", "limit": "5", "addressdetails": "1"}
    if country_codes:
        parameters["countrycodes"] = str(country_codes)
    request = urllib.request.Request(
        f"{endpoint}?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": "sevir-world-geospatial/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "geocoder_unavailable", "query": text, "reason": str(error), "provenance": {"source": endpoint}}
    candidates = []
    for item in payload if isinstance(payload, list) else []:
        try:
            candidate: dict[str, Any] = {
                "latitude": float(item["lat"]), "longitude": float(item["lon"]),
                "display_name": str(item.get("display_name") or text),
                "type": item.get("type"), "category": item.get("category"),
                "address": item.get("address", {}),
            }
            box = item.get("boundingbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                south, north, west, east = (float(value) for value in box)
                candidate["region"] = {"south": south, "west": west, "north": north, "east": east}
            candidates.append(candidate)
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        return {"status": "not_found", "query": text, "provenance": {"source": endpoint}}
    return {
        "status": "ok" if len(candidates) == 1 else "ambiguous", "query": text,
        **candidates[0], "candidates": candidates,
        "provenance": {"source": endpoint, "candidate_count": len(candidates)},
    }


__all__ = ["resolve_place"]
