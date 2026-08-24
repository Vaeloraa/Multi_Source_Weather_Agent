"""Fetch an Aurora-aligned ERA5 package for the selected SEVIR event.

The request is designed to return exactly 48 latitude by 96 longitude points
at 0.25 degrees.  It is separate from the first 49x81 ERA5 explanation cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = "R19080121067677"
OUT_DIR = ROOT / "data" / "era5_cases" / EVENT_ID
AREA = [40.75, -96.75, 29.0, -73.0]  # north, west, south, east -> 48x96 at 0.25°
GRID = "0.25/0.25"
YEAR = "2019"
MONTH = "08"
DAYS = ["01", "02"]
TIMES = [f"{hour:02d}:00" for hour in range(0, 24)]
PRESSURE_LEVELS = ["1000", "925", "850", "700", "600", "500", "400", "300", "250", "200", "150", "100", "50"]
SURFACE_VARIABLES = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
PRESSURE_VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]


def request(client: cdsapi.Client, dataset: str, variables: list[str], target: Path, **extra: object) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"cache_exists: {target}")
        return
    query: dict[str, object] = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [YEAR],
        "month": [MONTH],
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
    }
    query.update(extra)
    print(f"requesting: {dataset} -> {target.name}")
    client.retrieve(dataset, query, str(target))
    print(f"saved: {target} ({target.stat().st_size} bytes)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(timeout=60, retry_max=2)
    request(client, "reanalysis-era5-single-levels", SURFACE_VARIABLES, OUT_DIR / "aurora_aligned_surface_48x96.nc")
    request(client, "reanalysis-era5-pressure-levels", PRESSURE_VARIABLES, OUT_DIR / "aurora_aligned_pressure_48x96.nc", pressure_level=PRESSURE_LEVELS)
    request(client, "reanalysis-era5-single-levels", ["geopotential", "land_sea_mask", "soil_type"], OUT_DIR / "aurora_aligned_static_48x96.nc", day=["01"], time=["00:00"])
    manifest = {
        "event_id": EVENT_ID,
        "purpose": "Aurora input and subsequent ERA5 verification",
        "area_north_west_south_east": AREA,
        "grid": GRID,
        "expected_shape": {"latitude": 48, "longitude": 96},
        "surface_variables": SURFACE_VARIABLES,
        "pressure_variables": PRESSURE_VARIABLES,
        "pressure_levels_hpa": [int(level) for level in PRESSURE_LEVELS],
        "static_variables": ["geopotential", "land_sea_mask", "soil_type"],
        "files": ["aurora_aligned_surface_48x96.nc", "aurora_aligned_pressure_48x96.nc", "aurora_aligned_static_48x96.nc"],
        "input_time_rule": "Use two ERA5 times separated by six hours; validate against later ERA5 times.",
    }
    (OUT_DIR / "aurora_aligned_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {OUT_DIR / 'aurora_aligned_manifest.json'}")


if __name__ == "__main__":
    main()

