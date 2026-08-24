"""Build the persistent ERA5/Aurora cache for R19062708588472."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cdsapi


ROOT = Path(__file__).resolve().parents[1]
EVENT = "R19062708588472"
ANALYSIS = datetime(2019, 6, 27, 8, 58, 20, tzinfo=timezone.utc)
LAT, LON = 47.05431612570545, -99.97604419767778
AREA = [52.93, -111.85, 41.18, -88.10]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
PRESSURE = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]


def key() -> str:
    payload = {
        "event_id": EVENT, "analysis_time_utc": "2019-06-27T08:58:20Z",
        "latitude": LAT, "longitude": LON,
        "region": {"area_north_west_south_east": AREA},
        "variables": sorted(SURFACE + PRESSURE), "pressure_levels": [250, 500, 850],
        "history_before_hours": 24, "history_after_hours": 24,
        "forecast_leads_hours": [6, 12, 18, 24], "model": "era5+aurora",
        "model_version": "", "resolution": "0.25/0.25",
        "task": "foundation_request", "evidence_kind": "environment",
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def main() -> None:
    out = ROOT / "data" / "foundation_live_cache" / EVENT / key()
    out.mkdir(parents=True, exist_ok=True)
    start = datetime(2019, 6, 26, 8, tzinfo=timezone.utc)
    end = datetime(2019, 6, 28, 8, tzinfo=timezone.utc)
    days = sorted({(start + timedelta(hours=i)).strftime("%d") for i in range(49)})
    common = {
        "product_type": ["reanalysis"], "year": ["2019"], "month": ["06"],
        "day": days, "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA, "grid": "0.25/0.25", "data_format": "netcdf",
    }
    client = cdsapi.Client(timeout=120, retry_max=3)
    requests = [
        ("surface.nc", "reanalysis-era5-single-levels", dict(common, variable=SURFACE)),
        ("pressure.nc", "reanalysis-era5-pressure-levels", dict(common, variable=PRESSURE, pressure_level=[str(x) for x in LEVELS])),
        ("static.nc", "reanalysis-era5-single-levels", dict(common, variable=["geopotential", "land_sea_mask", "soil_type"], day=["01"], time=["00:00"])),
    ]
    for name, dataset, query in requests:
        target = out / name
        if target.exists() and target.stat().st_size > 100000:
            print(f"cache_exists {target}")
            continue
        # Never reuse/delete a possibly locked partial download left by an
        # interrupted CDS worker.  A unique destination makes retries safe on
        # Windows; only a fully completed file is moved to the canonical name.
        fresh = target.with_name(target.name + f".fresh.{uuid.uuid4().hex[:8]}.nc")
        print(f"request {dataset} {name}", flush=True)
        client.retrieve(dataset, query, str(fresh))
        fresh.replace(target)
        print(f"saved {target} {target.stat().st_size}", flush=True)

    ascii_dir = Path(tempfile.gettempdir()) / f"sevir_event472_{key()}"
    ascii_dir.mkdir(parents=True, exist_ok=True)
    for name in ("surface.nc", "pressure.nc", "static.nc"):
        shutil.copy2(out / name, ascii_dir / name)
    summary_script = ascii_dir / "summarize_era5_live.py"
    shutil.copy2(ROOT / "scripts" / "summarize_era5_live.py", summary_script)
    summary = out / "era5_target_timeseries.json"
    subprocess.run([
        sys.executable, str(summary_script), "--surface", str(ascii_dir / "surface.nc"),
        "--pressure", str(ascii_dir / "pressure.nc"), "--output", str(ascii_dir / summary.name),
        "--analysis-time", "2019-06-27T08:00:00Z", "--latitude", str(LAT), "--longitude", str(LON),
        "--before", "24", "--after", "24", "--levels", "850,500,250"], check=True)
    shutil.copy2(ascii_dir / summary.name, summary)
    (out / "request.json").write_text(json.dumps({
        "event_id": EVENT, "analysis_time_utc": ANALYSIS.isoformat().replace("+00:00", "Z"),
        "era5_time_utc": "2019-06-27T08:00:00Z", "latitude": LAT, "longitude": LON,
        "area_north_west_south_east": AREA, "history_before_hours": 24,
        "history_after_hours": 24, "forecast_leads_hours": [6, 12, 18, 24],
        "pressure_levels_hpa": [250, 500, 850], "aurora_pressure_levels_hpa": LEVELS,
        "canonical_request_key": key(), "status": "era5_ready",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "era5_ready", "artifact_dir": str(out), "summary": str(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
