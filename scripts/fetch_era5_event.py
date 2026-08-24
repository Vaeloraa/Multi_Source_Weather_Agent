"""Fetch the first ERA5 environmental supplement for one SEVIR event.

This is intentionally separate from Aurora.  It retrieves only the fields
needed to describe the event-scale environment and stores them as an
event-level cache that can later be reused by the environment tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]
EVENT_ID = "R19080121067677"
OUT_DIR = ROOT / "data" / "era5_cases" / EVENT_ID

# The selected SEVIR box is approximately 33.36--36.29 N, 87.24--82.49 W.
# This context box leaves room for the surrounding synoptic environment.
AREA = [41.0, -95.0, 29.0, -75.0]  # north, west, south, east
GRID = "0.25/0.25"
YEAR = "2019"
MONTH = "08"
DAYS = ["01", "02"]
TIMES = [f"{hour:02d}:00" for hour in range(12, 24)]

SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

PRESSURE_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

PRESSURE_LEVELS = ["250", "500", "850"]

# Aurora 0.25-degree pretrained uses the standard 13-level ERA5 atmosphere.
AURORA_PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300", "400",
    "500", "600", "700", "850", "925", "1000",
]


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

    request(
        client,
        "reanalysis-era5-single-levels",
        SURFACE_VARIABLES,
        OUT_DIR / "surface_2019-08-01_12-2019-08-02_23.nc",
    )
    request(
        client,
        "reanalysis-era5-pressure-levels",
        PRESSURE_VARIABLES,
        OUT_DIR / "pressure_2019-08-01_12-2019-08-02_23.nc",
        pressure_level=PRESSURE_LEVELS,
    )
    request(
        client,
        "reanalysis-era5-pressure-levels",
        PRESSURE_VARIABLES,
        OUT_DIR / "aurora_pressure_2019-08-01_12-2019-08-02_23.nc",
        pressure_level=AURORA_PRESSURE_LEVELS,
    )
    request(
        client,
        "reanalysis-era5-single-levels",
        ["geopotential", "land_sea_mask", "soil_type"],
        OUT_DIR / "static_era5_0.25deg.nc",
        day=["01"],
        time=["00:00"],
    )

    manifest = {
        "event_id": EVENT_ID,
        "sevir_analysis_time": "2019-08-01T21:06:40Z",
        "time_window_utc": {"start": "2019-08-01T12:00:00Z", "end": "2019-08-02T23:00:00Z"},
        "area_north_west_south_east": AREA,
        "grid": GRID,
        "surface_variables": SURFACE_VARIABLES,
        "pressure_variables": PRESSURE_VARIABLES,
            "pressure_levels_hpa": [int(level) for level in PRESSURE_LEVELS],
        "aurora_pressure_levels_hpa": [int(level) for level in AURORA_PRESSURE_LEVELS],
        "files": [
            "surface_2019-08-01_12-2019-08-02_23.nc",
            "pressure_2019-08-01_12-2019-08-02_23.nc",
            "aurora_pressure_2019-08-01_12-2019-08-02_23.nc",
            "static_era5_0.25deg.nc",
        ],
        "purpose": "ERA5 environmental supplement before Aurora integration",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
