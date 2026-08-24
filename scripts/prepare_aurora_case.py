"""Validate and describe the ERA5 inputs required by Aurora.

This is intentionally a preflight step: Aurora also requires static fields
(land-sea mask, soil type and static geopotential) that are not part of the
first environmental supplement request.  The script therefore refuses to
pretend that a forecast ran when those fields are absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xarray as xr


SURFACE = {"t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl"}
PRESSURE = {"t": "t", "u": "u", "v": "v", "q": "q", "z": "z"}
STATIC_REQUIRED = ("z", "slt", "lsm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with xr.open_dataset(args.surface) as surface, xr.open_dataset(args.pressure) as pressure:
        missing_surface = [name for name in SURFACE if name not in surface.data_vars]
        missing_pressure = [name for name in PRESSURE if name not in pressure.data_vars]
        lat = surface.coords.get("latitude")
        lon = surface.coords.get("longitude")
        times = surface.coords.get("valid_time", surface.coords.get("time"))
        levels = pressure.coords.get("pressure_level", pressure.coords.get("level"))
        static_names = []
        if args.static and args.static.is_file():
            with xr.open_dataset(args.static) as static:
                static_names = list(static.data_vars)
        # CDS returns pressure levels in descending atmospheric order.
        required_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        actual_levels = [int(value) for value in levels.values] if levels is not None else []
        static_ready = all(name in static_names for name in ("z", "lsm", "slt"))
        record = {
            "status": "ready_for_forecast" if not missing_surface and not missing_pressure and static_ready and actual_levels == required_levels else ("ready_for_static_fields" if not missing_surface and not missing_pressure else "invalid_inputs"),
            "surface_file": str(args.surface),
            "pressure_file": str(args.pressure),
            "surface_variables": {key: value for key, value in SURFACE.items() if key not in missing_surface},
            "pressure_variables": {key: value for key, value in PRESSURE.items() if key not in missing_pressure},
            "missing_surface_variables": missing_surface,
            "missing_pressure_variables": missing_pressure,
            "static_variables_required": list(STATIC_REQUIRED),
            "static_variables_available": static_names,
            "static_file": str(args.static) if args.static else None,
            "dimensions": {name: int(size) for name, size in surface.sizes.items()},
            "pressure_dimensions": {name: int(size) for name, size in pressure.sizes.items()},
            "latitude_decreasing": bool(lat is not None and lat.size > 1 and float(lat.values[0]) > float(lat.values[-1])),
            "longitude_count": int(lon.size) if lon is not None else None,
            "time_count": int(times.size) if times is not None else None,
            "pressure_levels": actual_levels,
            "required_pressure_levels": required_levels,
            "forecast_status": "ready_for_inference" if static_ready and actual_levels == required_levels else "blocked_input_contract",
            "interpretation_limit": "A valid Aurora forecast requires the static fields and the model checkpoint; this file is only an input contract.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
