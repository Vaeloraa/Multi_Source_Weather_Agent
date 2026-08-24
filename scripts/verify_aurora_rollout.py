"""Verify Aurora rollout outputs against subsequent ERA5 fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - truth.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-index", type=int, default=18)
    parser.add_argument("--latitude", type=float, default=34.82941969338152)
    parser.add_argument("--longitude", type=float, default=-84.86671230653173)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast = np.load(args.forecast)
    if "lead_hours" not in forecast.files:
        lead_hours = sorted(int(key.split("_")[1]) for key in forecast.files if key.startswith("lead_") and key.endswith("_2t"))
        normalized = {"lead_hours": np.asarray(lead_hours, dtype=np.int32)}
        for variable in ("2t", "10u", "10v", "msl", "atmos_t", "atmos_u", "atmos_v", "atmos_q", "atmos_z"):
            normalized[variable] = np.stack([forecast[f"lead_{lead:03d}_{variable}"][0, 0] for lead in lead_hours], axis=0)[None, ...]
        forecast = normalized
    else:
        lead_hours = [int(v) for v in forecast["lead_hours"]]
    with xr.open_dataset(args.surface) as surface, xr.open_dataset(args.pressure) as pressure:
        truth_indices = [args.input_index + hour for hour in lead_hours]
        if max(truth_indices) >= surface.sizes["valid_time"]:
            raise ValueError("forecast leads exceed the available ERA5 validation period")
        target_lat, target_lon = args.latitude, args.longitude
        lat_index = int(np.abs(surface.latitude.values - target_lat).argmin())
        lon_index = int(np.abs(surface.longitude.values - target_lon).argmin())
        region_metrics: dict[str, list[dict[str, float]]] = {}
        target_series: dict[str, list[float]] = {"2m_temperature_c": [], "mslp_hpa": [], "wind_speed_ms": []}
        truth_series: dict[str, list[float]] = {key: [] for key in target_series}
        for lead_position, (lead, truth_index) in enumerate(zip(lead_hours, truth_indices)):
            pred_t = forecast["2t"][0, lead_position,  :, :]
            pred_u = forecast["10u"][0, lead_position, :, :]
            pred_v = forecast["10v"][0, lead_position, :, :]
            pred_m = forecast["msl"][0, lead_position, :, :]
            truth_t = surface.t2m.values[truth_index] 
            truth_u = surface.u10.values[truth_index]
            truth_v = surface.v10.values[truth_index]
            truth_m = surface.msl.values[truth_index]
            pred_wind = np.hypot(pred_u, pred_v)
            truth_wind = np.hypot(truth_u, truth_v)
            region_metrics.setdefault("2m_temperature_c", []).append(_metrics(pred_t - 273.15, truth_t - 273.15))
            region_metrics.setdefault("mslp_hpa", []).append(_metrics(pred_m / 100.0, truth_m / 100.0))
            region_metrics.setdefault("wind_speed_ms", []).append(_metrics(pred_wind, truth_wind))
            target_series["2m_temperature_c"].append(float(pred_t[lat_index, lon_index] - 273.15))
            target_series["mslp_hpa"].append(float(pred_m[lat_index, lon_index] / 100.0))
            target_series["wind_speed_ms"].append(float(pred_wind[lat_index, lon_index]))
            truth_series["2m_temperature_c"].append(float(truth_t[lat_index, lon_index] - 273.15))
            truth_series["mslp_hpa"].append(float(truth_m[lat_index, lon_index] / 100.0))
            truth_series["wind_speed_ms"].append(float(truth_wind[lat_index, lon_index]))
        # Add all 13-level atmospheric field scores.
        for name, era_name in (("t", "t"), ("u", "u"), ("v", "v"), ("q", "q"), ("z", "z")):
            prediction = forecast[f"atmos_{name}"][0]
            level_metrics = {}
            for level_position, level in enumerate(pressure.pressure_level.values):
                level_metrics[str(int(level))] = []
                for lead_position, truth_index in enumerate(truth_indices):
                    truth = getattr(pressure, era_name).values[truth_index, level_position]
                    level_metrics[str(int(level))].append(_metrics(prediction[lead_position, level_position], truth))
            region_metrics[f"pressure_{name}"] = level_metrics
        summary = {
            "status": "ok",
            "verification_status": "full_domain_multi_lead",
            "lead_hours": lead_hours,
            "truth_indices": truth_indices,
            "target_grid_point": {"latitude": float(surface.latitude.values[lat_index]), "longitude": float(surface.longitude.values[lon_index])},
            "surface_region_metrics": region_metrics,
            "target_point_forecast": target_series,
            "target_point_era5": truth_series,
            "interpretation_limit": "Verification compares Aurora large-scale fields with subsequent ERA5; it does not validate local VIL prediction.",
        }
        output_json = args.output_dir / "aurora_vs_era5_verification.json"
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        hours = np.asarray(lead_hours)
        figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
        labels = [("2m_temperature_c", "2 m temperature (°C)"), ("mslp_hpa", "MSLP (hPa)"), ("wind_speed_ms", "10 m wind speed (m/s)")]
        for axis, (key, label) in zip(axes, labels):
            axis.plot(hours, target_series[key], marker="o", label="Aurora")
            axis.plot(hours, truth_series[key], marker="o", label="ERA5")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
        axes[-1].set_xlabel("Lead time (hours)")
        figure.suptitle("Aurora vs subsequent ERA5 at target point")
        figure.tight_layout()
        figure.savefig(args.output_dir / "aurora_vs_era5_timeseries.png", dpi=150)
        plt.close(figure)
        # Three surface error maps at +6h.
        lead_position = 0
        error_panels = [
            (forecast["2t"][0, lead_position] - 273.15) - (surface.t2m.values[truth_indices[0]] - 273.15),
            forecast["msl"][0, lead_position] / 100.0 - surface.msl.values[truth_indices[0]] / 100.0,
            np.hypot(forecast["10u"][0, lead_position], forecast["10v"][0, lead_position]) - np.hypot(surface.u10.values[truth_indices[0]], surface.v10.values[truth_indices[0]]),
        ]
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for axis, values, title in zip(axes, error_panels, ("2 m temperature error", "MSLP error", "10 m wind speed error")):
            image = axis.imshow(values, origin="upper", cmap="coolwarm")
            axis.set_title(title)
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle("Aurora - ERA5 error at +6h")
        figure.tight_layout()
        figure.savefig(args.output_dir / "aurora_error_maps.png", dpi=150)
        plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
