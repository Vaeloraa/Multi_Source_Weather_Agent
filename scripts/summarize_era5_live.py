"""Extract compact target-point evidence from live ERA5 NetCDF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _time_series(data: xr.DataArray) -> np.ndarray:
    value = data
    for dimension in value.dims:
        if dimension != "valid_time":
            value = value.isel({dimension: 0})
    return np.asarray(value.values, dtype=np.float64).reshape(-1)


def _rounded(values: np.ndarray, digits: int = 4) -> list[float | None]:
    return [None if not np.isfinite(value) else round(float(value), digits) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-time", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--before", type=int, default=6)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--levels", default="850,500,250")
    args = parser.parse_args()

    target_time = np.datetime64(str(args.analysis_time).replace("Z", ""), "s")
    window_start = target_time - np.timedelta64(args.before, "h")
    window_end = target_time + np.timedelta64(args.after, "h")
    requested_levels = [int(value) for value in args.levels.split(",") if value.strip()]

    with xr.open_dataset(args.surface) as surface, xr.open_dataset(args.pressure) as pressure:
        surface_point = surface.sel(latitude=args.latitude, longitude=args.longitude, method="nearest")
        surface_times = surface.valid_time.values.astype("datetime64[s]")
        mask = (surface_times >= window_start) & (surface_times <= window_end)
        indices = np.nonzero(mask)[0]
        if not indices.size:
            raise ValueError("ERA5 files do not cover the requested target-point time window")
        selected_times = surface_times[indices]
        time_strings = [np.datetime_as_string(value, unit="s") + "Z" for value in selected_times]

        t2m = _time_series(surface_point["t2m"])[indices] - 273.15
        u10 = _time_series(surface_point["u10"])[indices]
        v10 = _time_series(surface_point["v10"])[indices]
        msl = _time_series(surface_point["msl"])[indices] / 100.0
        wind10 = np.sqrt(u10 ** 2 + v10 ** 2)
        event_index = int(np.argmin(np.abs(selected_times.astype("int64") - target_time.astype("int64"))))

        pressure_point = pressure.sel(latitude=args.latitude, longitude=args.longitude, method="nearest")
        pressure_times = pressure.valid_time.values.astype("datetime64[s]")
        pressure_index = {int(value.astype("int64")): index for index, value in enumerate(pressure_times)}
        matched_pressure_indices = np.asarray([pressure_index[int(value.astype("int64"))] for value in selected_times], dtype=int)
        levels = {}
        for requested in requested_levels:
            level_point = pressure_point.sel(pressure_level=requested, method="nearest")
            temperature = _time_series(level_point["t"])[matched_pressure_indices] - 273.15
            u_value = _time_series(level_point["u"])[matched_pressure_indices]
            v_value = _time_series(level_point["v"])[matched_pressure_indices]
            humidity = _time_series(level_point["q"])[matched_pressure_indices] * 1000.0
            height = _time_series(level_point["z"])[matched_pressure_indices] / 9.80665
            levels[str(requested)] = {
                "temperature_c": _rounded(temperature),
                "u_m_s": _rounded(u_value),
                "v_m_s": _rounded(v_value),
                "wind_m_s": _rounded(np.sqrt(u_value ** 2 + v_value ** 2)),
                "specific_humidity_g_kg": _rounded(humidity),
                "geopotential_height_m": _rounded(height, 2),
            }

        result = {
            "status": "ok",
            "source": "ERA5 live CDS retrieval",
            "requested_target": {"latitude": args.latitude, "longitude": args.longitude},
            "selected_target": {
                "latitude": float(surface_point.latitude.values),
                "longitude": float(surface_point.longitude.values),
            },
            "analysis_time_utc": np.datetime_as_string(target_time, unit="s") + "Z",
            "coverage": {
                "start_utc": time_strings[0],
                "end_utc": time_strings[-1],
                "time_count": len(time_strings),
                "history_before_hours": args.before,
                "history_after_hours": args.after,
                "event_nearest_index": event_index,
                "event_nearest_time_utc": time_strings[event_index],
            },
            "times_utc": time_strings,
            "surface": {
                "t2m_c": _rounded(t2m),
                "u10_m_s": _rounded(u10),
                "v10_m_s": _rounded(v10),
                "wind10_m_s": _rounded(wind10),
                "mslp_hpa": _rounded(msl),
            },
            "pressure_levels": levels,
            "endpoint_change": {
                "t2m_c": round(float(t2m[-1] - t2m[0]), 4),
                "wind10_m_s": round(float(wind10[-1] - wind10[0]), 4),
                "mslp_hpa": round(float(msl[-1] - msl[0]), 4),
            },
            "interpretation_limit": "ERA5 is retrospective reanalysis; it describes environmental fields and does not directly predict local VIL.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "time_count": len(time_strings)}))


if __name__ == "__main__":
    main()
