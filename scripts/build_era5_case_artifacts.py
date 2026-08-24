"""Build three readable ERA5 artifacts for the selected SEVIR case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


EVENT_TIME = np.datetime64("2019-08-01T21:06:40")
TARGET_LAT = 34.829419693381515
TARGET_LON = -84.86671230653173
G = 9.80665


def _nearest_time(ds: xr.Dataset) -> np.datetime64:
    values = ds["valid_time"].values
    return values[np.argmin(np.abs(values - EVENT_TIME))]


def _field(ax: plt.Axes, lat: np.ndarray, lon: np.ndarray, values: np.ndarray, title: str, cmap: str = "viridis") -> None:
    mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap)
    ax.scatter([TARGET_LON], [TARGET_LAT], c="red", s=24, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, shrink=0.82)


def build(surface_path: Path, pressure_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(surface_path) as surface, xr.open_dataset(pressure_path) as pressure:
        valid_time = _nearest_time(surface)
        s = surface.sel(valid_time=valid_time)
        p = pressure.sel(valid_time=valid_time)

        t2m = s["t2m"] - 273.15
        wind10 = np.hypot(s["u10"], s["v10"])
        msl = s["msl"] / 100.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        _field(axes[0], surface.latitude, surface.longitude, t2m.values, f"2 m temperature ({valid_time})", "coolwarm")
        _field(axes[1], surface.latitude, surface.longitude, wind10.values, "10 m wind speed (m/s)", "magma")
        _field(axes[2], surface.latitude, surface.longitude, msl.values, "mean sea-level pressure (hPa)", "viridis")
        fig.savefig(output_dir / "era5_surface_environment.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(11, 13), constrained_layout=True)
        for row, level in enumerate([850, 500, 250]):
            level_data = p.sel(pressure_level=level)
            height = level_data["z"] / G
            wind = np.hypot(level_data["u"], level_data["v"])
            _field(axes[row, 0], pressure.latitude, pressure.longitude, height.values, f"{level} hPa geopotential height (m)", "viridis")
            _field(axes[row, 1], pressure.latitude, pressure.longitude, wind.values, f"{level} hPa wind speed (m/s)", "magma")
        fig.savefig(output_dir / "era5_pressure_environment.png", dpi=160)
        plt.close(fig)

        point_surface = surface.sel(latitude=TARGET_LAT, longitude=TARGET_LON, method="nearest")
        point_pressure = pressure.sel(latitude=TARGET_LAT, longitude=TARGET_LON, pressure_level=850, method="nearest")
        times = surface.valid_time.values
        t2m_series = (point_surface["t2m"] - 273.15).values
        wind_series = np.hypot(point_surface["u10"], point_surface["v10"]).values
        msl_series = (point_surface["msl"] / 100.0).values
        q850_series = point_pressure["q"].values * 1000.0

        timeseries_payload = {
            "event_id": "R19080121067677",
            "event_time_utc": str(EVENT_TIME),
            "target_point": {"latitude": TARGET_LAT, "longitude": TARGET_LON},
            "nearest_grid_point": {
                "latitude": float(point_surface.latitude.values),
                "longitude": float(point_surface.longitude.values),
            },
            "source": "ERA5_reanalysis",
            "sampling_intervals_hours": [int(value) for value in np.diff(times).astype("timedelta64[h]").astype(int)],
            "has_time_gaps": bool(np.any(np.diff(times).astype("timedelta64[h]").astype(int) > 1)),
            "variables": {
                "time_utc": [str(value) for value in times],
                "2m_temperature_c": [float(value) for value in t2m_series],
                "10m_wind_speed_ms": [float(value) for value in wind_series],
                "mslp_hpa": [float(value) for value in msl_series],
                "850hpa_specific_humidity_gkg": [float(value) for value in q850_series],
            },
            "interpretation_limit": "ERA5 is retrospective environmental context, not a local VIL forecast.",
        }
        (output_dir / "era5_target_timeseries.json").write_text(
            json.dumps(timeseries_payload, indent=2), encoding="utf-8"
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        axes[0, 0].plot(times, t2m_series, marker="o")
        axes[0, 0].set_title("2 m temperature (°C)")
        axes[0, 1].plot(times, wind_series, marker="o", color="darkorange")
        axes[0, 1].set_title("10 m wind speed (m/s)")
        axes[1, 0].plot(times, msl_series, marker="o", color="seagreen")
        axes[1, 0].set_title("mean sea-level pressure (hPa)")
        axes[1, 1].plot(times, q850_series, marker="o", color="purple")
        axes[1, 1].set_title("850 hPa specific humidity (g/kg)")
        for ax in axes.flat:
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"ERA5 target-point time series: {TARGET_LAT:.2f}N, {TARGET_LON:.2f}E")
        fig.savefig(output_dir / "era5_target_timeseries.png", dpi=160)
        plt.close(fig)

        summary = {
            "event_id": "R19080121067677",
            "event_time": str(EVENT_TIME),
            "era5_nearest_time": str(valid_time),
            "target_point": {"latitude": TARGET_LAT, "longitude": TARGET_LON},
            "nearest_grid_point": {
                "latitude": float(point_surface.latitude.values),
                "longitude": float(point_surface.longitude.values),
            },
            "artifacts": [
                "era5_surface_environment.png",
                "era5_pressure_environment.png",
                "era5_target_timeseries.png",
                "era5_target_timeseries.json",
            ],
            "interpretation_limit": "ERA5 is retrospective environmental context, not a local VIL forecast.",
        }
        (output_dir / "era5_environment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.surface, args.pressure, args.output)
    print(f"artifacts_written: {args.output}")


if __name__ == "__main__":
    main()
