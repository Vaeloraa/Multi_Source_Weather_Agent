"""Raw ERA5-reanalysis/Aurora field loading for the public capability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..era5.backend import fetch_era5_live, normalize_live_request
from .backend import run_aurora_live


def _utc_timestamp(value: str) -> float:
    """Convert an ISO-8601 UTC value to a Unix timestamp.

    Parameters
    ----------
    value : str
        ISO-8601 date-time. A trailing ``Z`` and timezone-naive values are
        interpreted as UTC.

    Returns
    -------
    float
        Seconds since the Unix epoch.
    """
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def regional_environment_fields(
    event_id: str | None,
    initialization_time_utc: str,
    latitude: float | None,
    longitude: float | None,
    region: dict[str, float] | None,
    forecast_leads: list[int],
    history_before_hours: int,
    history_after_hours: int,
    data_policy: str,
) -> dict[str, Any]:
    """Load aligned raw regional fields without selecting a hazard or score.

    Parameters
    ----------
    event_id : str, optional
        Event identity retained as request provenance.
    initialization_time_utc : str
        Aurora initialization time in ISO-8601 UTC format.
    latitude, longitude : float, optional
        Optional point around which a default regional request is built.
    region : dict, optional
        Explicit ``south``, ``west``, ``north`` and ``east`` bounds.
    forecast_leads : list of int
        Aurora forecast lead hours to load.
    history_before_hours, history_after_hours : int
        ERA5/ERA5T reanalysis context interval around the analysis time.
    data_policy : str
        Data policy such as ``"cached"`` or live-capable access.

    Returns
    -------
    dict
        Coordinates, aligned ERA5 fields, Aurora fields and cache provenance.
    """

    request = normalize_live_request(
        analysis_time_utc=initialization_time_utc,
        latitude=latitude,
        longitude=longitude,
        region=region,
        forecast_leads=forecast_leads,
        history_before_hours=history_before_hours,
        history_after_hours=history_after_hours,
        event_id=str(event_id or ""),
    )
    cache_only = str(data_policy).lower() == "cached"
    request["access_mode"] = "cache_only" if cache_only else "cache_then_remote"
    requested_region = dict(region) if isinstance(region, dict) else {
        "latitude": latitude, "longitude": longitude,
    }
    north, west, south, east = request["area_north_west_south_east"]
    effective_region = {
        "south": float(south), "west": float(west),
        "north": float(north), "east": float(east),
    }
    fetched = fetch_era5_live(request, cache_only=cache_only)
    if fetched.get("status") != "ok":
        return {
            "status": "era5_unavailable",
            "reason": fetched.get("reason") or fetched.get("error_code"),
            "era5_temporality": fetched.get("era5_temporality")
            or request.get("era5_temporality"),
        }
    fetched_files = fetched.get("files") if isinstance(fetched.get("files"), dict) else {}
    cached_surface = Path(str(fetched_files.get("surface", "")))
    artifact_root = cached_surface.parent if cached_surface.is_file() else Path(str(request["artifact_dir"]))
    aurora = run_aurora_live(request, artifact_dir=artifact_root, cache_only=cache_only)
    if aurora.get("status") != "ok":
        return {"status": "aurora_unavailable", "reason": aurora.get("reason") or aurora.get("error_code")}

    files = fetched_files
    root = artifact_root
    surface_path = Path(str(files.get("surface") or root / "surface.nc"))
    pressure_path = Path(str(files.get("pressure") or root / "pressure.nc"))
    forecast_path = Path(str(aurora.get("output") or root / "aurora_forecast.npz"))
    summary_path = root / "aurora_forecast.json"
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
        aligned_time = str(request.get("era5_time_utc") or initialization_time_utc)
        time_index = int(np.argmin(np.abs(valid_times - _utc_timestamp(aligned_time))))
        era5_surface = {
            "t2m_c": np.asarray(surface_file["t2m"][time_index], dtype=np.float32) - 273.15,
            "u10_m_s": np.asarray(surface_file["u10"][time_index], dtype=np.float32),
            "v10_m_s": np.asarray(surface_file["v10"][time_index], dtype=np.float32),
            "mslp_hpa": np.asarray(surface_file["msl"][time_index], dtype=np.float32) / 100.0,
        }

    era5_pressure: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(pressure_path, "r") as pressure_file:
        levels = [int(value) for value in np.asarray(pressure_file["pressure_level"]).tolist()]
        for level in levels:
            level_index = levels.index(level)
            era5_pressure[str(level)] = {
                "temperature_c": np.asarray(pressure_file["t"][time_index, level_index], dtype=np.float32) - 273.15,
                "u_m_s": np.asarray(pressure_file["u"][time_index, level_index], dtype=np.float32),
                "v_m_s": np.asarray(pressure_file["v"][time_index, level_index], dtype=np.float32),
                "specific_humidity_g_kg": np.asarray(pressure_file["q"][time_index, level_index], dtype=np.float32) * 1000.0,
                "geopotential_height_m": np.asarray(pressure_file["z"][time_index, level_index], dtype=np.float32) / 9.80665,
            }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_levels = [int(value) for value in summary.get("pressure_levels", [])]
    completed = [int(value) for value in summary.get("completed_leads_hours", [])]
    selected_leads = [int(value) for value in forecast_leads if int(value) in completed]
    aurora_fields: dict[str, Any] = {}
    with np.load(forecast_path, allow_pickle=False) as payload:
        for lead in selected_leads:
            prefix = f"lead_{lead:03d}_"
            pressure_fields: dict[str, dict[str, np.ndarray]] = {}
            for level in model_levels:
                level_index = model_levels.index(level)
                pressure_fields[str(level)] = {
                    "temperature_c": np.asarray(payload[prefix + "atmos_t"][0, 0, level_index], dtype=np.float32) - 273.15,
                    "u_m_s": np.asarray(payload[prefix + "atmos_u"][0, 0, level_index], dtype=np.float32),
                    "v_m_s": np.asarray(payload[prefix + "atmos_v"][0, 0, level_index], dtype=np.float32),
                    "specific_humidity_g_kg": np.asarray(payload[prefix + "atmos_q"][0, 0, level_index], dtype=np.float32) * 1000.0,
                    "geopotential_height_m": np.asarray(payload[prefix + "atmos_z"][0, 0, level_index], dtype=np.float32) / 9.80665,
                }
            aurora_fields[str(lead)] = {
                "surface": {
                    "t2m_c": np.asarray(payload[prefix + "2t"][0, 0], dtype=np.float32) - 273.15,
                    "u10_m_s": np.asarray(payload[prefix + "10u"][0, 0], dtype=np.float32),
                    "v10_m_s": np.asarray(payload[prefix + "10v"][0, 0], dtype=np.float32),
                    "mslp_hpa": np.asarray(payload[prefix + "msl"][0, 0], dtype=np.float32) / 100.0,
                },
                "pressure_levels": pressure_fields,
            }

    return {
        "status": "ok",
        "forecast_id": str(request["request_key"]),
        "initialization_time_utc": str(request.get("aurora_initialization_time_utc") or initialization_time_utc),
        "latitude": latitudes,
        "longitude": longitudes,
        "era5": {
            "surface": era5_surface,
            "pressure_levels": era5_pressure,
            "temporality": dict(request.get("era5_temporality") or {}),
            "is_real_time": False,
            "access_mode": str(request.get("access_mode") or "cache_then_remote"),
        },
        "aurora": {"lead_hours": selected_leads, "fields": aurora_fields},
        "requested_region": requested_region,
        "effective_region": effective_region,
        "requested_leads_hours": [int(value) for value in forecast_leads],
        "missing_leads_hours": [int(value) for value in forecast_leads if int(value) not in selected_leads],
        "cache_hit": bool(fetched.get("cache_hit", False)) and bool(aurora.get("cache_hit", False)),
    }
