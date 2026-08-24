"""Run a single Aurora pretrained rollout for the selected ERA5 case."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata, rollout


def _prepare(values: np.ndarray, index: int, flip_lat: bool) -> torch.Tensor:
    # ERA5 files are [time, (level), latitude, longitude]. Aurora expects a
    # batch dimension and decreasing latitude.
    selected = values[[index - 6, index]]
    if flip_lat:
        selected = selected[..., ::-1, :]
    return torch.from_numpy(np.ascontiguousarray(selected[None]))


def _static(values: np.ndarray, flip_lat: bool) -> torch.Tensor:
    """Reduce invariant ERA5 fields to [latitude, longitude]."""
    array = np.asarray(values)
    while array.ndim > 2:
        array = array[0]
    if flip_lat:
        array = array[::-1, :]
    return torch.from_numpy(np.ascontiguousarray(array))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=6, help="latest input index; the other input is six hours earlier")
    parser.add_argument("--analysis-time", default="", help="ERA5 valid time used as the forecast initialization")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--leads", default="6", help="comma-separated forecast lead hours; multiples of six")
    args = parser.parse_args()

    with xr.open_dataset(args.surface) as surface, xr.open_dataset(args.pressure) as atmos, xr.open_dataset(args.static) as static:
        height = int(surface.sizes["latitude"])
        width = int(surface.sizes["longitude"])
        if height % 4 or width % 4:
            raise ValueError(f"Aurora input grid must be divisible by patch size 4; got {height}x{width}. Query an aligned ERA5 area instead of cropping.")
        usable_height, usable_width = height, width
        if args.analysis_time:
            valid_times = surface.valid_time.values.astype("datetime64[s]")
            target_time = np.datetime64(str(args.analysis_time).replace("Z", ""), "s")
            exact = np.nonzero(valid_times == target_time)[0]
            if exact.size:
                i = int(exact[0])
            else:
                i = int(np.argmin(np.abs(valid_times.astype("int64") - target_time.astype("int64"))))
        else:
            i = int(args.index)
        if i < 6 or i >= surface.sizes["valid_time"]:
            raise ValueError("index must select two ERA5 times separated by six hours")
        lat = surface.latitude.values
        lon = surface.longitude.values
        flip_lat = bool(lat.size > 1 and float(lat[0]) < float(lat[-1]))
        levels = tuple(int(v) for v in atmos.pressure_level.values)
        batch = Batch(
            surf_vars={
                "2t": _prepare(surface.t2m.values, i, flip_lat),
                "10u": _prepare(surface.u10.values, i, flip_lat),
                "10v": _prepare(surface.v10.values, i, flip_lat),
                "msl": _prepare(surface.msl.values, i, flip_lat),
            },
            static_vars={
                "z": _static(static.z.values, flip_lat),
                "slt": _static(static.slt.values, flip_lat),
                "lsm": _static(static.lsm.values, flip_lat),
            },
            atmos_vars={
                "t": _prepare(atmos.t.values, i, flip_lat),
                "u": _prepare(atmos.u.values, i, flip_lat),
                "v": _prepare(atmos.v.values, i, flip_lat),
                "q": _prepare(atmos.q.values, i, flip_lat),
                "z": _prepare(atmos.z.values, i, flip_lat),
            },
            metadata=Metadata(
                lat=torch.from_numpy(np.ascontiguousarray(lat[::-1] if flip_lat else lat)),
                # Aurora requires longitudes in [0, 360), while CDS regional
                # requests commonly return the -180..180 convention.
                lon=torch.from_numpy(np.ascontiguousarray(lon % 360.0)),
                time=(surface.valid_time.values.astype("datetime64[s]").tolist()[i],),
                atmos_levels=levels,
            ),
        )
        model = AuroraPretrained()
        model.load_checkpoint()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        requested_leads = sorted({int(value) for value in str(args.leads).split(",") if value.strip()})
        if not requested_leads or any(value <= 0 or value % 6 for value in requested_leads):
            raise ValueError("Aurora leads must be positive multiples of six hours")
        requested_set = set(requested_leads)
        max_steps = max(requested_leads) // 6
        output_lat = np.ascontiguousarray(lat[::-1] if flip_lat else lat)
        output_lon = np.ascontiguousarray(lon % 360.0)
        target_lat = float(args.latitude) if args.latitude is not None else float(output_lat[len(output_lat) // 2])
        target_lon = (float(args.longitude) % 360.0) if args.longitude is not None else float(output_lon[len(output_lon) // 2])
        lat_index = int(np.argmin(np.abs(output_lat.astype(np.float64) - target_lat)))
        lon_distance = np.abs(((output_lon.astype(np.float64) - target_lon + 180.0) % 360.0) - 180.0)
        lon_index = int(np.argmin(lon_distance))
        arrays = {}
        lead_summaries = {}
        with torch.inference_mode():
            for step, pred in enumerate(rollout(model, batch, steps=max_steps), start=1):
                lead = step * 6
                if lead not in requested_set:
                    continue
                surface_arrays = {name: value.detach().cpu().numpy() for name, value in pred.surf_vars.items()}
                atmos_arrays = {name: value.detach().cpu().numpy() for name, value in pred.atmos_vars.items()}
                for name, value in surface_arrays.items():
                    arrays[f"lead_{lead:03d}_{name}"] = value
                for name, value in atmos_arrays.items():
                    arrays[f"lead_{lead:03d}_atmos_{name}"] = value
                if lead == requested_leads[0]:
                    arrays.update(surface_arrays)
                    arrays.update({f"atmos_{name}": value for name, value in atmos_arrays.items()})
                point = {
                    "time_utc": str(pred.metadata.time[0]),
                    "latitude": float(output_lat[lat_index]),
                    "longitude": float(output_lon[lon_index]),
                    "t2m_c": float(surface_arrays["2t"][0, 0, lat_index, lon_index] - 273.15),
                    "u10_m_s": float(surface_arrays["10u"][0, 0, lat_index, lon_index]),
                    "v10_m_s": float(surface_arrays["10v"][0, 0, lat_index, lon_index]),
                    "mslp_hpa": float(surface_arrays["msl"][0, 0, lat_index, lon_index] / 100.0),
                    "pressure_levels": {},
                }
                point["wind10_m_s"] = float((point["u10_m_s"] ** 2 + point["v10_m_s"] ** 2) ** 0.5)
                for level in (850, 500, 250):
                    if level not in levels:
                        continue
                    level_index = levels.index(level)
                    u_value = float(atmos_arrays["u"][0, 0, level_index, lat_index, lon_index])
                    v_value = float(atmos_arrays["v"][0, 0, level_index, lat_index, lon_index])
                    point["pressure_levels"][str(level)] = {
                        "temperature_c": float(atmos_arrays["t"][0, 0, level_index, lat_index, lon_index] - 273.15),
                        "u_m_s": u_value,
                        "v_m_s": v_value,
                        "wind_m_s": float((u_value ** 2 + v_value ** 2) ** 0.5),
                        "specific_humidity_g_kg": float(atmos_arrays["q"][0, 0, level_index, lat_index, lon_index] * 1000.0),
                        "geopotential_height_m": float(atmos_arrays["z"][0, 0, level_index, lat_index, lon_index] / 9.80665),
                    }
                lead_summaries[str(lead)] = point
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        summary = {
            "status": "ok",
            "model": "AuroraPretrained",
            "event_input_time": str(surface.valid_time.values[i]),
            "device": str(device),
            "pressure_levels": list(levels),
            "input_grid_shape": [int(usable_height), int(usable_width)],
            "output": str(args.output),
            "completed_leads_hours": requested_leads,
            "target_point_forecasts": lead_summaries,
            "surface_output_shapes": {name: list(value.shape) for name, value in arrays.items() if "atmos_" not in name},
            "atmos_output_shapes": {name: list(value.shape) for name, value in arrays.items() if "atmos_" in name},
            "interpretation_limit": "This is a large-scale environmental forecast; it is not a local VIL forecast.",
        }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
