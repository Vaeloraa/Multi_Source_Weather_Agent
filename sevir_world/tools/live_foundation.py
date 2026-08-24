from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..data_access.live_foundation import fetch_era5_live, normalize_live_request, run_aurora_live


def _request_defaults(
    event_id: str | None,
    analysis_time_utc: str | None,
    latitude: float | None,
    longitude: float | None,
    region: dict[str, Any] | None,
    forecast_leads: list[int] | None,
    history_before_hours: int | None,
    history_after_hours: int | None,
) -> tuple[str | None, str | None, float | None, float | None, dict[str, Any] | None, list[int] | None, int | None, int | None]:
    """Fill omitted tool arguments from the request-scoped live context.

    Agent code is allowed to call a live tool without repeating values
    already established by the HTTP preflight.  The worker receives these
    values through an explicit environment allowlist; cached requests never
    populate it, so this cannot silently bind an unrelated event.
    """
    if event_id is None:
        event_id = os.getenv("SEVIR_LIVE_EVENT_ID")
    if analysis_time_utc is None:
        analysis_time_utc = os.getenv("SEVIR_LIVE_ANALYSIS_TIME_UTC")
    if latitude is None and os.getenv("SEVIR_LIVE_LATITUDE"):
        latitude = float(os.environ["SEVIR_LIVE_LATITUDE"])
    if longitude is None and os.getenv("SEVIR_LIVE_LONGITUDE"):
        longitude = float(os.environ["SEVIR_LIVE_LONGITUDE"])
    if region is None and os.getenv("SEVIR_LIVE_REGION_JSON"):
        value = json.loads(os.environ["SEVIR_LIVE_REGION_JSON"])
        region = value if isinstance(value, dict) else None
    if forecast_leads is None and os.getenv("SEVIR_LIVE_FORECAST_LEADS"):
        forecast_leads = [int(item) for item in os.environ["SEVIR_LIVE_FORECAST_LEADS"].split(",") if item.strip()]
    if history_before_hours is None and os.getenv("SEVIR_LIVE_HISTORY_BEFORE_HOURS"):
        history_before_hours = int(os.environ["SEVIR_LIVE_HISTORY_BEFORE_HOURS"])
    if history_after_hours is None and os.getenv("SEVIR_LIVE_HISTORY_AFTER_HOURS"):
        history_after_hours = int(os.environ["SEVIR_LIVE_HISTORY_AFTER_HOURS"])
    return event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours


def live_preflight(
    event_id: str | None = None,
    analysis_time_utc: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    region: dict[str, Any] | None = None,
    variables: list[str] | None = None,
    pressure_variables: list[str] | None = None,
    pressure_levels: list[int] | None = None,
    forecast_leads: list[int] | None = None,
    history_before_hours: int | None = None,
    history_after_hours: int | None = None,
) -> dict[str, Any]:
    """Normalize a live time/location request without reading a local case."""
    event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours = _request_defaults(
        event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours
    )
    if not analysis_time_utc:
        raise ValueError("analysis_time_utc is required for live foundation analysis")
    return normalize_live_request(
        analysis_time_utc=analysis_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        variables=variables,
        pressure_variables=pressure_variables,
        pressure_levels=pressure_levels,
        forecast_leads=forecast_leads,
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id=str(event_id or ""),
    )


def fetch_era5(
    event_id: str | None = None,
    analysis_time_utc: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    region: dict[str, Any] | None = None,
    variables: list[str] | None = None,
    pressure_variables: list[str] | None = None,
    pressure_levels: list[int] | None = None,
    forecast_leads: list[int] | None = None,
    history_before_hours: int | None = None,
    history_after_hours: int | None = None,
) -> dict[str, Any]:
    """Fetch ERA5 on demand for a live request and return its manifest."""
    event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours = _request_defaults(
        event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours
    )
    if not analysis_time_utc:
        raise ValueError("analysis_time_utc is required for live foundation analysis")
    request = normalize_live_request(
        analysis_time_utc=analysis_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        variables=variables,
        pressure_variables=pressure_variables,
        pressure_levels=pressure_levels,
        forecast_leads=forecast_leads,
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id=str(event_id or ""),
    )
    return fetch_era5_live(request)


def _event_snapshot_from_timeseries(
    fetched: dict[str, Any],
) -> dict[str, Any]:
    """Project a cached ERA5 series onto its declared event-time sample."""
    series = fetched.get("target_timeseries")
    if not isinstance(series, dict) or series.get("status") != "ok":
        return {
            "status": "unavailable",
            "error_code": "era5_target_timeseries_missing",
            "cache_hit": bool(fetched.get("cache_hit", False)),
        }
    coverage = series.get("coverage") if isinstance(series.get("coverage"), dict) else {}
    index = int(coverage.get("event_nearest_index", 0))
    times = list(series.get("times_utc", []))

    def at(values: Any) -> Any:
        return values[index] if isinstance(values, list) and 0 <= index < len(values) else None

    surface_source = series.get("surface") if isinstance(series.get("surface"), dict) else {}
    surface = {key: at(surface_source.get(key)) for key in (
        "t2m_c", "u10_m_s", "v10_m_s", "wind10_m_s", "mslp_hpa"
    )}
    levels: dict[str, Any] = {}
    pressure_source = (
        series.get("pressure_levels")
        if isinstance(series.get("pressure_levels"), dict)
        else {}
    )
    for level in ("850", "500", "250"):
        source = pressure_source.get(level)
        if not isinstance(source, dict):
            continue
        levels[level] = {key: at(source.get(key)) for key in (
            "temperature_c", "u_m_s", "v_m_s", "wind_m_s",
            "specific_humidity_g_kg", "geopotential_height_m",
        )}
    return {
        "status": "ok",
        "cache_hit": bool(fetched.get("cache_hit", False)),
        "event_id": fetched.get("event_id"),
        "request_key": fetched.get("request_key"),
        "analysis_time_utc": fetched.get("analysis_time_utc"),
        "era5_time_utc": times[index] if 0 <= index < len(times) else fetched.get("era5_time_utc"),
        "time_offset_minutes": fetched.get("time_offset_minutes"),
        "requested_target": series.get("requested_target"),
        "selected_target": series.get("selected_target"),
        "surface": surface,
        "pressure_levels": levels,
        "units": {
            "temperature": "degC",
            "wind": "m/s",
            "specific_humidity": "g/kg",
            "mean_sea_level_pressure": "hPa",
            "geopotential_height": "m",
        },
        "source": series.get("source", "ERA5 reanalysis"),
        "evidence_role": "historical_reanalysis_event_snapshot",
        "supported_conclusions": [
            "event-time grid-point temperature, humidity, pressure, wind, and geopotential values",
            "vertical ordering of the directly returned pressure-level values",
            "cautious environmental interpretation whose numerical or profile-comparison basis is stated",
        ],
        "unsupported_without_additional_diagnostics": [
            "climatological high/low categories",
            "moisture transport or dynamical lifting",
            "convective triggering, storm type, surface rainfall, or hazard",
        ],
        "unsupported_without_regional_fields": [
            "fronts or pressure-centre presence or control",
        ],
        "interpretation_limit": (
            "This is one ERA5 grid-point snapshot, not a direct measurement. It supports "
            "transparent, cautious interpretation of the returned profile, but not spatial "
            "systems, temporal trends, a unique triggering mechanism, or guaranteed impacts."
        ),
    }


def era5_snapshot(**kwargs: Any) -> dict[str, Any]:
    """Return one compact ERA5 event-time snapshot from the live/cached request."""
    return _event_snapshot_from_timeseries(fetch_era5(**kwargs))


def era5_timeseries(**kwargs: Any) -> dict[str, Any]:
    """Return the ERA5 target-point historical series with its declared coverage."""
    fetched = fetch_era5(**kwargs)
    series = fetched.get("target_timeseries")
    if not isinstance(series, dict):
        return {
            "status": "unavailable",
            "error_code": "era5_target_timeseries_missing",
            "cache_hit": bool(fetched.get("cache_hit", False)),
        }
    result = dict(series)
    result["cache_hit"] = bool(fetched.get("cache_hit", False))
    result["event_id"] = fetched.get("event_id")
    result["request_key"] = fetched.get("request_key")
    result["evidence_role"] = "historical_reanalysis_timeseries"
    return result


def forecast_aurora(
    event_id: str | None = None,
    analysis_time_utc: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    region: dict[str, Any] | None = None,
    variables: list[str] | None = None,
    pressure_variables: list[str] | None = None,
    pressure_levels: list[int] | None = None,
    forecast_leads: list[int] | None = None,
    history_before_hours: int | None = None,
    history_after_hours: int | None = None,
) -> dict[str, Any]:
    """Fetch live ERA5 inputs when needed and run AuroraPretrained."""
    event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours = _request_defaults(
        event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours
    )
    if not analysis_time_utc:
        raise ValueError("analysis_time_utc is required for live foundation analysis")
    request = normalize_live_request(
        analysis_time_utc=analysis_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        variables=variables,
        pressure_variables=pressure_variables,
        pressure_levels=pressure_levels,
        forecast_leads=forecast_leads,
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id=str(event_id or ""),
    )
    fetched = fetch_era5_live(request)
    if fetched.get("status") != "ok":
        return {"status": "era5_unavailable", "era5": fetched, "request": request}
    return run_aurora_live(request)


def _utc_timestamp(value: str) -> float:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def region_fields(
    event_id: str | None = None,
    analysis_time_utc: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    region: dict[str, Any] | None = None,
    forecast_leads: list[int] | None = None,
    history_before_hours: int | None = None,
    history_after_hours: int | None = None,
) -> dict[str, Any]:
    """Return cached ERA5/Aurora regional grids for Agent-selected screening.

    The adapter selects no thresholds and labels no point as hazardous. It
    standardizes coordinates, units and dimensions so generated code can
    derive a transparent question-specific score or comparison.
    """
    event_id, analysis_time_utc, latitude, longitude, region, forecast_leads, history_before_hours, history_after_hours = _request_defaults(
        event_id, analysis_time_utc, latitude, longitude, region, forecast_leads,
        history_before_hours, history_after_hours,
    )
    if not analysis_time_utc:
        raise ValueError("analysis_time_utc is required for regional foundation fields")
    request = normalize_live_request(
        analysis_time_utc=analysis_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        forecast_leads=forecast_leads,
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id=str(event_id or ""),
    )
    fetched = fetch_era5_live(request)
    if fetched.get("status") != "ok":
        return {"status": "era5_unavailable", "era5": fetched, "request": request}
    aurora = run_aurora_live(request)
    if aurora.get("status") != "ok":
        return {"status": "aurora_unavailable", "aurora": aurora, "request": request}

    files = fetched.get("files") if isinstance(fetched.get("files"), dict) else {}
    root = Path(str(request["artifact_dir"]))
    surface_path = Path(str(files.get("surface") or root / "surface.nc"))
    pressure_path = Path(str(files.get("pressure") or root / "pressure.nc"))
    forecast_path = Path(str(aurora.get("output") or root / "aurora_forecast.npz"))
    summary_path = root / "aurora_forecast.json"
    # A demo may contain a model output initialized earlier than the event so
    # its requested lead is valid at the event time. This is still a raw model
    # product, not a cached screening result. Select the unique aligned pair
    # and leave all scores, thresholds and matching to generated Agent code.
    aligned_pairs = [
        (candidate, candidate.with_suffix(".json"))
        for candidate in root.glob("aurora_forecast_aligned_*.npz")
        if candidate.with_suffix(".json").is_file()
    ]
    if len(aligned_pairs) == 1:
        forecast_path, summary_path = aligned_pairs[0]
    if not all(path.is_file() for path in (surface_path, pressure_path, forecast_path, summary_path)):
        raise RuntimeError("cached regional ERA5/Aurora artifacts are incomplete")

    with h5py.File(surface_path, "r") as surface_file:
        latitudes = np.asarray(surface_file["latitude"], dtype=np.float32)
        longitudes = np.asarray(surface_file["longitude"], dtype=np.float32)
        valid_times = np.asarray(surface_file["valid_time"], dtype=np.int64)
        aligned_time = str(request.get("era5_time_utc") or analysis_time_utc)
        time_index = int(np.argmin(np.abs(valid_times - _utc_timestamp(aligned_time))))
        era5_surface = {
            "t2m_c": np.asarray(surface_file["t2m"][time_index], dtype=np.float32) - 273.15,
            "u10_m_s": np.asarray(surface_file["u10"][time_index], dtype=np.float32),
            "v10_m_s": np.asarray(surface_file["v10"][time_index], dtype=np.float32),
            "mslp_hpa": np.asarray(surface_file["msl"][time_index], dtype=np.float32) / 100.0,
        }
    era5_surface["wind10_m_s"] = np.sqrt(
        era5_surface["u10_m_s"] ** 2 + era5_surface["v10_m_s"] ** 2
    ).astype(np.float32)

    selected_levels = [850, 500, 250]
    era5_pressure: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(pressure_path, "r") as pressure_file:
        levels = [int(value) for value in np.asarray(pressure_file["pressure_level"]).tolist()]
        for level in selected_levels:
            level_index = levels.index(level)
            u = np.asarray(pressure_file["u"][time_index, level_index], dtype=np.float32)
            v = np.asarray(pressure_file["v"][time_index, level_index], dtype=np.float32)
            era5_pressure[str(level)] = {
                "temperature_c": np.asarray(pressure_file["t"][time_index, level_index], dtype=np.float32) - 273.15,
                "u_m_s": u,
                "v_m_s": v,
                "wind_m_s": np.sqrt(u ** 2 + v ** 2).astype(np.float32),
                "specific_humidity_g_kg": np.asarray(pressure_file["q"][time_index, level_index], dtype=np.float32) * 1000.0,
                "geopotential_height_m": np.asarray(pressure_file["z"][time_index, level_index], dtype=np.float32) / 9.80665,
            }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_levels = [int(value) for value in summary.get("pressure_levels", [])]
    completed = [int(value) for value in summary.get("completed_leads_hours", [])]
    requested = [int(value) for value in (forecast_leads or completed)]
    selected_leads = [value for value in requested if value in completed]
    aurora_fields: dict[str, Any] = {}
    with np.load(forecast_path, allow_pickle=False) as payload:
        for lead in selected_leads:
            prefix = f"lead_{lead:03d}_"
            u10 = np.asarray(payload[prefix + "10u"][0, 0], dtype=np.float32)
            v10 = np.asarray(payload[prefix + "10v"][0, 0], dtype=np.float32)
            pressure_fields: dict[str, dict[str, np.ndarray]] = {}
            for level in selected_levels:
                level_index = model_levels.index(level)
                u = np.asarray(payload[prefix + "atmos_u"][0, 0, level_index], dtype=np.float32)
                v = np.asarray(payload[prefix + "atmos_v"][0, 0, level_index], dtype=np.float32)
                pressure_fields[str(level)] = {
                    "temperature_c": np.asarray(payload[prefix + "atmos_t"][0, 0, level_index], dtype=np.float32) - 273.15,
                    "u_m_s": u,
                    "v_m_s": v,
                    "wind_m_s": np.sqrt(u ** 2 + v ** 2).astype(np.float32),
                    "specific_humidity_g_kg": np.asarray(payload[prefix + "atmos_q"][0, 0, level_index], dtype=np.float32) * 1000.0,
                    "geopotential_height_m": np.asarray(payload[prefix + "atmos_z"][0, 0, level_index], dtype=np.float32) / 9.80665,
                }
            aurora_fields[str(lead)] = {
                "surface": {
                    "t2m_c": np.asarray(payload[prefix + "2t"][0, 0], dtype=np.float32) - 273.15,
                    "u10_m_s": u10,
                    "v10_m_s": v10,
                    "wind10_m_s": np.sqrt(u10 ** 2 + v10 ** 2).astype(np.float32),
                    "mslp_hpa": np.asarray(payload[prefix + "msl"][0, 0], dtype=np.float32) / 100.0,
                },
                "pressure_levels": pressure_fields,
            }

    return {
        "status": "ok",
        "task": "foundation_region_fields",
        "event_id": event_id,
        "analysis_time_utc": analysis_time_utc,
        "era5_time_utc": request.get("era5_time_utc"),
        "latitude": latitudes,
        "longitude": longitudes,
        "dimensions": ["latitude", "longitude"],
        "era5": {"surface": era5_surface, "pressure_levels": era5_pressure},
        "aurora": {"lead_hours": selected_leads, "fields": aurora_fields},
        "cache_hit": bool(fetched.get("cache_hit", False)) and bool(aurora.get("cache_hit", False)),
        "screening_rule": "No score, threshold, category, or candidate was preselected by the adapter.",
        "interpretation_limit": "Regional fields support relative candidate screening; candidates are not confirmed local convection or hazards.",
    }
