"""On-demand ERA5 request normalization and retrieval for live Agent runs.

This module deliberately does not depend on an event-level case directory.  A
live request is identified by its time, spatial target, variables and model
configuration; any downloaded files are session artifacts keyed by that
request signature.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from typing import Any

from ...cache import atomic_json
from .temporality import classify_era5_request
from ...cache.identity import request_identity


DEFAULT_SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
DEFAULT_PRESSURE_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
DEFAULT_AURORA_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
AURORA_PIPELINE_VERSION = "era5-aurora-bundle-v2"
AURORA_PACKAGE_VERSION = os.getenv(
    "SEVIR_AURORA_PACKAGE_VERSION", "2.0.1"
).strip()
AURORA_MODEL_VERSION = os.getenv(
    "SEVIR_AURORA_MODEL_VERSION",
    f"AuroraPretrained:microsoft-aurora=={AURORA_PACKAGE_VERSION}:"
    "default-checkpoint:worker-v2",
).strip()


def _netcdf_readable(path: Path, request_key: str) -> bool:
    """Validate NetCDF/HDF5 integrity without copying a cached file.

    The HTTP worker already ships with a compatible h5py build.  Opening the
    file and enumerating its root objects checks the HDF5 superblock and object
    table without reading the meteorological arrays.  The foreign-interpreter
    fallback is retained for lightweight deployments which do not ship h5py,
    but it is intentionally not the normal cache-hit path.
    """
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        import h5py  # type: ignore

        with h5py.File(path, "r") as handle:
            return bool(list(handle.keys()))
    except ImportError:
        pass
    except (OSError, ValueError):
        return False
    stage = Path(tempfile.gettempdir()) / f"sevir_netcdf_check_{str(request_key or 'request')[:32]}"
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / path.name
    try:
        if not staged.is_file() or staged.stat().st_size != path.stat().st_size:
            shutil.copyfile(path, staged)
        probe = (
            "import h5py,sys; "
            "f=h5py.File(sys.argv[1],'r'); f.visit(lambda _: None); f.close()"
        )
        probe_environment = os.environ.copy()
        # The HTTP service prepends its pure-Python vendor directory so the
        # lightweight interpreter can read SEVIR.  Passing that PYTHONPATH to
        # the separate Aurora Python 3.10 process shadows its compatible h5py
        # binary and makes valid NetCDF files look corrupt.
        probe_environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [_foundation_python(), "-c", probe, str(staged)],
            capture_output=True, text=True, timeout=60, env=probe_environment,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            staged.unlink(missing_ok=True)
            stage.rmdir()
        except OSError:
            pass


def _foundation_python() -> str:
    configured = os.getenv("AURORA_PYTHON") or os.getenv("CASCAST_PYTHON")
    if configured:
        return configured
    sibling = Path(__file__).resolve().parents[4] / "CasCast" / ".venv" / "Scripts" / "python.exe"
    return str(sibling) if sibling.is_file() else sys.executable


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("analysis_time_utc is required for on-demand foundation analysis")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _aligned_hour(value: datetime) -> tuple[str, int]:
    aligned = value.replace(minute=0, second=0, microsecond=0)
    delta_minutes = int(round((value - aligned).total_seconds() / 60.0))
    return aligned.strftime("%Y-%m-%dT%H:00:00Z"), delta_minutes


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _area_from_target(latitude: Any, longitude: Any, region: dict[str, Any] | None) -> list[float]:
    if isinstance(region, dict) and all(key in region for key in ("north", "west", "south", "east")):
        north = _number(region["north"], "region.north")
        west = _number(region["west"], "region.west")
        south = _number(region["south"], "region.south")
        east = _number(region["east"], "region.east")
    else:
        lat = _number(latitude, "latitude")
        lon = _number(longitude, "longitude")
        # At 0.25 degrees, Aurora's patch size requires point counts that are
        # divisible by four.  Use 48 latitude points and 96 longitude points
        # while keeping the requested point inside the regional box.
        north, south = lat + 6.0, lat - 5.75
        west, east = lon - 12.0, lon + 11.75
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("region must satisfy -90<=south<north<=90 and -180<=west<east<=180")
    # Normalize any supplied box to an Aurora-compatible 0.25 degree grid.
    # Small event bounding boxes are expanded around their center; larger
    # boxes are rounded up to a multiple-of-four point count.
    center_lat, center_lon = (north + south) / 2.0, (east + west) / 2.0
    lat_points = max(48, int(round((north - south) / 0.25)) + 1)
    lon_points = max(96, int(round((east - west) / 0.25)) + 1)
    lat_points += (-lat_points) % 4
    lon_points += (-lon_points) % 4
    lat_span, lon_span = (lat_points - 1) * 0.25, (lon_points - 1) * 0.25
    north, south = center_lat + lat_span / 2.0, center_lat - lat_span / 2.0
    west, east = center_lon - lon_span / 2.0, center_lon + lon_span / 2.0
    if south < -90 or north > 90 or west < -180 or east > 180:
        raise ValueError("region cannot be expanded to an Aurora-compatible grid within WGS84 bounds")
    return [round(north, 2), round(west, 2), round(south, 2), round(east, 2)]


def _era5_area(latitude: Any, longitude: Any, region: dict[str, Any] | None) -> list[float]:
    """Preserve ERA5 bounds, including global and polar regions, without padding."""
    if region is not None:
        from ..geospatial.contracts import BoundingBox
        box = BoundingBox.from_value(region)
        return [box.north, box.west, box.south, box.east]
    if latitude is None and longitude is None:
        return [90.0, -180.0, -90.0, 180.0]
    from ..geospatial.contracts import GeoPoint
    point = GeoPoint(float(latitude), float(longitude))
    # A point request selects the closest regular ERA5 grid point.
    lat = round(point.latitude * 4) / 4
    lon = round(point.longitude * 4) / 4
    return [lat, lon, lat, lon]


def normalize_live_request(
    analysis_time_utc: str,
    latitude: float | None = None,
    longitude: float | None = None,
    region: dict[str, Any] | None = None,
    variables: list[str] | None = None,
    pressure_variables: list[str] | None = None,
    pressure_levels: list[int] | None = None,
    forecast_leads: list[int] | None = None,
    history_before_hours: int | None = None,
    history_after_hours: int | None = None,
    cache_root: str | Path | None = None,
    event_id: str = "",
    model_version: str = "",
    model_contract: str = "aurora",
) -> dict[str, Any]:
    parsed = _parse_time(analysis_time_utc)
    aligned, delta_minutes = _aligned_hour(parsed)
    if model_contract != "data_selection" and region is None and (latitude is None or longitude is None):
        raise ValueError("on-demand foundation analysis requires latitude/longitude or region")
    area = (_era5_area if model_contract == "data_selection" else _area_from_target)(latitude, longitude, region)
    levels = sorted({int(item) for item in (
        [250, 500, 850] if pressure_levels is None else pressure_levels
    )}, reverse=True)
    if any(level <= 0 for level in levels):
        raise ValueError("pressure_levels must contain positive hPa values")
    lead_values = [6, 12, 18, 24] if forecast_leads is None else forecast_leads
    leads = sorted({int(item) for item in lead_values})
    if any(lead <= 0 for lead in leads):
        raise ValueError("forecast_leads must contain positive hours")
    before_hours = 6 if history_before_hours is None else int(history_before_hours)
    after_hours = 0 if history_after_hours is None else int(history_after_hours)
    minimum_before = 0 if model_contract == "data_selection" else 6
    valid_window = (before_hours >= 0 and after_hours >= 0) if model_contract == "data_selection" else (
        minimum_before <= before_hours <= 72 and 0 <= after_hours <= 72
    )
    if not valid_window:
        raise ValueError(
            ("history windows must be non-negative hours " if model_contract == "data_selection" else
             f"history windows must satisfy {minimum_before}<=before<=72 and 0<=after<=72 hours ") +
            f"for {model_contract}; received before={before_hours}, after={after_hours}"
        )
    request = {
        "analysis_time_utc": parsed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "era5_time_utc": aligned,
        "time_offset_minutes": delta_minutes,
        "latitude": None if latitude is None else _number(latitude, "latitude"),
        "longitude": None if longitude is None else _number(longitude, "longitude"),
        "area_north_west_south_east": area,
        "grid": "0.25/0.25",
        "surface_variables": list(DEFAULT_SURFACE_VARIABLES if variables is None else variables),
        "pressure_variables": list(
            DEFAULT_PRESSURE_VARIABLES if pressure_variables is None else pressure_variables
        ),
        "pressure_levels_hpa": levels,
        "aurora_pressure_levels_hpa": (
            list(DEFAULT_AURORA_LEVELS) if model_contract == "aurora" else list(levels)
        ),
        "forecast_leads_hours": leads,
        "input_window_hours": 6,
        "history_before_hours": before_hours,
        "history_after_hours": after_hours,
        "data_policy": "live",
        # ``live`` is the access policy (a cache miss may use remote I/O), not
        # the scientific temporality of ERA5.  The fields below are the
        # authoritative distinction presented to the Agent and UI.
        "access_mode": "cache_then_remote",
        "is_real_time_data": False,
        "event_id": str(event_id or "").strip().upper(),
        "model_version": str(model_version or AURORA_MODEL_VERSION),
        "model_contract": str(model_contract),
    }
    # The bundle includes history, target-point summaries and forecast fields,
    # so every dependency participates in its one cache key.
    # A resolved place often contributes both an explicit regional bounding
    # box and its centre point. Regional ERA5/Aurora fields are determined by
    # the box; allowing the redundant geocoder point into the physical bundle
    # identity makes the same grid miss cache depending only on whether the
    # geocoder happened to resolve. Point identity remains significant for a
    # point-only request.
    explicit_region = region is not None or (latitude is None and longitude is None)
    target_latitude = (
        (float(area[0]) + float(area[2])) / 2.0
        if explicit_region
        else float(request["latitude"])
    )
    target_longitude = (
        (float(area[1]) + float(area[3])) / 2.0
        if explicit_region
        else float(request["longitude"])
    )
    bundle_identity = request_identity(
        analysis_time_utc=request["era5_time_utc"],
        latitude=target_latitude,
        longitude=target_longitude,
        region={"area_north_west_south_east": area},
        variables=request["surface_variables"] + request["pressure_variables"],
        pressure_levels=sorted(set(
            request["aurora_pressure_levels_hpa"] + request["pressure_levels_hpa"]
        )),
        history_before_hours=before_hours,
        history_after_hours=after_hours,
        forecast_leads_hours=leads,
        model="era5+aurora" if model_contract == "aurora" else "era5",
        model_version=request["model_version"],
        resolution=request["grid"],
        task=AURORA_PIPELINE_VERSION if model_contract == "aurora" else "era5-data-selection-v1",
        evidence_kind="era5_raw+aurora_field" if model_contract == "aurora" else "era5_raw",
    )
    request["aurora_initialization_time_utc"] = request["era5_time_utc"]
    request["era5_temporality"] = classify_era5_request(
        request["era5_time_utc"],
        history_before_hours=before_hours,
        history_after_hours=after_hours,
        forecast_leads_hours=leads,
    )
    request["data_temporality"] = request["era5_temporality"]["data_class"]
    request["cache_parameters"] = bundle_identity["canonical"]
    request["request_key"] = bundle_identity["request_key"]
    root = Path(cache_root) if cache_root else Path(__file__).resolve().parents[3] / "data" / "foundation_live_cache"
    capability = "aurora" if model_contract == "aurora" else "era5"
    request["artifact_dir"] = str(root / capability / request["request_key"])
    request["status"] = "normalized"
    return request


def cds_client():
    """Load CDS in the service interpreter or the configured model environment."""
    try:
        import cdsapi
    except ImportError:
        candidates = []
        configured = os.getenv("CASCAST_PYTHON", "")
        if configured:
            candidates.append(Path(configured).resolve().parent.parent / "Lib" / "site-packages")
        candidates.append(Path(__file__).resolve().parents[4] / "CasCast" / ".venv" / "Lib" / "site-packages")
        for candidate in candidates:
            if (candidate / "cdsapi").exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        import cdsapi
    return cdsapi.Client(timeout=60, retry_max=2)


def _fetch_selected(request: dict[str, Any], output_dir: Path, cache_only: bool) -> dict[str, Any]:
    """Partition the exact ERA5-only window; completed calendar parts are reusable."""
    from .cds import retrieve_data
    reference = _parse_time(request["era5_time_utc"])
    start = reference - timedelta(hours=request["history_before_hours"])
    end = reference + timedelta(hours=request["history_after_hours"])
    # Group days sharing the same requested hours, within one calendar month.
    # This avoids accidental extra dates/hours at month and year boundaries.
    groups: dict[tuple[int, int, tuple[str, ...]], list[str]] = {}
    cursor = start.replace(hour=0)
    while cursor <= end:
        hours = tuple(f"{hour:02d}:00" for hour in range(24) if start <= cursor.replace(hour=hour) <= end)
        groups.setdefault((cursor.year, cursor.month, hours), []).append(f"{cursor.day:02d}")
        cursor += timedelta(days=1)
    files: dict[str, Any] = {}
    file_variables: dict[str, dict[str, list[str]]] = {}
    hits = []

    def cached(dataset, query):
        try:
            return retrieve_data(dataset, query, data_policy="cached", cache_root=output_dir / "cds")
        except RuntimeError as error:
            if not str(error).startswith("cache_only_miss:"):
                raise
            return None

    def keep(group, raw, names, paths):
        paths.extend(raw["files"])
        file_variables[group].update({path: list(names) for path in raw["files"]})
        hits.append(raw["cache_hit"])

    for group, variables, dataset in (
        ("surface", request["surface_variables"], "reanalysis-era5-single-levels"),
        ("pressure", request["pressure_variables"], "reanalysis-era5-pressure-levels"),
    ):
        if not variables:
            continue
        paths = []
        file_variables[group] = {}
        for (year, month, hours), days in groups.items():
            query = {
                "product_type": ["reanalysis"], "variable": sorted(variables),
                "year": [str(year)], "month": [f"{month:02d}"], "day": days, "time": list(hours),
                "area": request["area_north_west_south_east"], "grid": request["grid"],
                "data_format": "netcdf", "download_format": "unarchived",
            }
            if group == "pressure":
                query["pressure_level"] = [str(level) for level in request["pressure_levels_hpa"]]
            batch = cached(dataset, query)
            if batch is not None:
                keep(group, batch, variables, paths)
                continue
            # Preserve previously downloaded single-variable parts. Only missing
            # variables share the next CDS job; do not redownload cached data.
            missing = []
            for variable in sorted(variables):
                single = cached(dataset, {**query, "variable": [variable]})
                if single is None:
                    missing.append(variable)
                else:
                    keep(group, single, [variable], paths)
            if missing:
                raw = retrieve_data(dataset, {**query, "variable": missing},
                                    data_policy="cached" if cache_only else "live", cache_root=output_dir / "cds")
                keep(group, raw, missing, paths)
        files[group] = paths[0] if len(paths) == 1 else paths
    return {**request, "status": "ok", "files": files, "file_variables": file_variables, "cache_hit": all(hits)}


def fetch_era5_live(
    request: dict[str, Any],
    *,
    artifact_dir: str | Path | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Fetch ERA5 reanalysis on demand through CDS and write a manifest.

    "Live" denotes permission to perform remote retrieval after a cache miss; the returned
    data always carries an ERA5/ERA5T reanalysis temporality record and is
    never labelled as a real-time observation or forecast.
    """
    if not isinstance(request, dict) or request.get("data_policy") != "live":
        raise ValueError("fetch_era5_live requires a normalized live request")
    temporality = request.get("era5_temporality")
    if not isinstance(temporality, dict):
        temporality = classify_era5_request(
            str(request.get("era5_time_utc") or request.get("analysis_time_utc") or ""),
            history_before_hours=int(request.get("history_before_hours", 0) or 0),
            history_after_hours=int(request.get("history_after_hours", 0) or 0),
            forecast_leads_hours=request.get("forecast_leads_hours", []),
        )
        request["era5_temporality"] = temporality
        request["data_temporality"] = temporality["data_class"]
    if request.get("model_contract") != "data_selection" and temporality.get("availability_status") == "not_yet_available":
        return {
            "status": "era5_not_available_yet",
            "error_code": "era5_reanalysis_release_lag",
            "reason": (
                "ERA5 is reanalysis, not a real-time feed; the newest requested "
                "valid time is inside the ERA5T publication lag."
            ),
            "era5_temporality": temporality,
            "request": request,
        }
    output_dir = Path(artifact_dir or request.get("artifact_dir", ""))
    expected_key = str(request["request_key"])
    needs_surface = bool(request.get("surface_variables"))
    needs_pressure = bool(request.get("pressure_variables"))
    needs_static = request.get("model_contract") == "aurora"
    required_names = [
        *( ["surface.nc"] if needs_surface else [] ),
        *( ["pressure.nc"] if needs_pressure else [] ),
        *( ["static.nc"] if needs_static else [] ),
    ]
    manifest_path = output_dir / "live_manifest.json"
    try:
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached_manifest = None
    identity_matches = (
        isinstance(cached_manifest, dict)
        and str(cached_manifest.get("request_key", "")) == expected_key
    )
    if (
        identity_matches and cached_manifest.get("status") == "ok"
        and all(_netcdf_readable(output_dir / name, expected_key) for name in required_names)
    ):
        # Files may come from another study; request facts belong to this call.
        manifest = dict(request)
        manifest["cache_hit"] = True
        manifest["status"] = "ok"
        manifest["cache_compatibility"] = "exact"
        # Verified files in the request-key directory are authoritative.
        manifest["artifact_dir"] = str(output_dir)
        manifest["files"] = {
            Path(name).stem: str(output_dir / name) for name in required_names
        }
        return manifest
    if request.get("model_contract") == "data_selection":
        return _fetch_selected(request, output_dir, cache_only)
    if cache_only or os.getenv("SEVIR_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            f"cache_only_miss: refusing to download ERA5 for request {expected_key}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = cds_client()
    except ImportError:
        return {
            "status": "dependency_missing", "error_code": "cdsapi_not_installed",
            "message": "cdsapi is required for on-demand ERA5 reanalysis retrieval",
            "request": request,
        }
    timestamp = _parse_time(str(request["era5_time_utc"]))
    before_hours = int(request.get("history_before_hours", 6))
    after_hours = int(request.get("history_after_hours", 0))
    start = timestamp.replace(minute=0, second=0, microsecond=0) - __import__("datetime").timedelta(hours=before_hours)
    end = timestamp.replace(minute=0, second=0, microsecond=0) + __import__("datetime").timedelta(hours=after_hours)
    total_hours = int((end - start).total_seconds() // 3600)
    input_times = [start + __import__("datetime").timedelta(hours=i) for i in range(total_hours + 1)]
    days = sorted({item.strftime("%d") for item in input_times})
    months = sorted({item.strftime("%m") for item in input_times})
    years = sorted({item.strftime("%Y") for item in input_times})
    hours = sorted({item.strftime("%H:%M") for item in input_times})
    area = request["area_north_west_south_east"]
    grid = request.get("grid", "0.25/0.25")
    surface = output_dir / "surface.nc"
    pressure = output_dir / "pressure.nc"
    static = output_dir / "static.nc"
    # A complete manifest binds files to their request. Unbound or mismatched
    # files must be downloaded again even when they are readable HDF5.
    invalid_files: set[Path] = set()
    required_paths = [
        *( [surface] if needs_surface else [] ),
        *( [pressure] if needs_pressure else [] ),
        *( [static] if needs_static else [] ),
    ]
    for candidate in required_paths:
        if candidate.exists() and (
            not identity_matches or not _netcdf_readable(candidate, expected_key)
        ):
            invalid_files.add(candidate)
            try:
                candidate.unlink()
            except OSError:
                pass
    common = {
        "product_type": ["reanalysis"],
        "year": years, "month": months, "day": days, "time": hours,
        "area": area, "grid": grid, "data_format": "netcdf",
    }
    def retrieve_checked(dataset: str, query: dict[str, Any], target: Path) -> Path:
        destination = target
        if target in invalid_files:
            destination = target.with_name(target.name + ".fresh.nc")
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        client.retrieve(dataset, query, str(destination))
        if not _netcdf_readable(destination, expected_key):
            raise RuntimeError(f"CDS returned an invalid or truncated NetCDF file: {destination.name}")
        if destination != target:
            try:
                os.replace(destination, target)
                destination = target
            except OSError:
                # A previous worker may still hold the old path.  Keep the
                # verified fresh file and use it for this request instead of
                # silently falling back to the corrupt cache entry.
                pass
        return destination

    if needs_surface and (not surface.exists() or surface in invalid_files):
        query = dict(common, variable=request["surface_variables"])
        surface = retrieve_checked("reanalysis-era5-single-levels", query, surface)
    if needs_pressure and (not pressure.exists() or pressure in invalid_files):
        query = dict(common, variable=request["pressure_variables"], pressure_level=[str(x) for x in request["aurora_pressure_levels_hpa"]])
        pressure = retrieve_checked("reanalysis-era5-pressure-levels", query, pressure)
    if needs_static and (not static.exists() or static in invalid_files):
        query = dict(common, variable=["geopotential", "land_sea_mask", "soil_type"], day=["01"], time=["00:00"])
        static = retrieve_checked("reanalysis-era5-single-levels", query, static)
    files = {
        **({"surface": str(surface)} if needs_surface else {}),
        **({"pressure": str(pressure)} if needs_pressure else {}),
        **({"static": str(static)} if needs_static else {}),
    }
    manifest = dict(request)
    manifest.update({"status": "ok", "files": files})
    atomic_json(output_dir / "live_manifest.json", manifest)
    return manifest
