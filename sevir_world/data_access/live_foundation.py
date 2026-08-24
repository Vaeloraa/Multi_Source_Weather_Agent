"""On-demand ERA5 request normalization and retrieval for live Agent runs.

This module deliberately does not depend on an event-level case directory.  A
live request is identified by its time, spatial target, variables and model
configuration; any downloaded files are session artifacts keyed by that
request signature.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import ctypes
import shutil
import tempfile
from typing import Any

from .cache_layers import write_manifest, read_manifest
from .evidence_bundle import make_evidence
from .request_identity import request_identity


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


def _subprocess_path(path: str | Path) -> str:
    """Return an ASCII-safe Windows short path for foreign Python processes."""
    value = str(path)
    if os.name != "nt" or value.isascii():
        return value
    try:
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        source = ctypes.c_wchar_p(value)
        size = get_short(source, None, 0)
        if size:
            buffer = ctypes.create_unicode_buffer(size + 1)
            written = get_short(source, buffer, size + 1)
            if written:
                return buffer.value
    except (AttributeError, OSError):
        pass
    return value


def _ascii_stage(request_key: str, *paths: Path) -> tuple[Path, dict[Path, Path]]:
    """Copy files to an ASCII-only temp directory for foreign Python workers."""
    stage = Path(tempfile.gettempdir()) / f"sevir_foundation_{str(request_key or 'request')[:32]}"
    stage.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    for source in paths:
        target = stage / source.name
        if source.is_file() and (not target.exists() or target.stat().st_size != source.stat().st_size):
            shutil.copy2(source, target)
        mapping[source] = target
    return stage, mapping


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
    sibling = Path(__file__).resolve().parents[3] / "CasCast" / ".venv" / "Scripts" / "python.exe"
    return str(sibling) if sibling.is_file() else sys.executable


def _parse_time(value: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("analysis_time_utc is required for live foundation analysis")
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
) -> dict[str, Any]:
    parsed = _parse_time(analysis_time_utc)
    aligned, delta_minutes = _aligned_hour(parsed)
    if region is None and (latitude is None or longitude is None):
        raise ValueError("live foundation analysis requires latitude/longitude or region")
    area = _area_from_target(latitude, longitude, region)
    levels = sorted({int(item) for item in (pressure_levels or [250, 500, 850])}, reverse=True)
    if any(level <= 0 for level in levels):
        raise ValueError("pressure_levels must contain positive hPa values")
    leads = sorted({int(item) for item in (forecast_leads or [6, 12, 18, 24])})
    if any(lead <= 0 for lead in leads):
        raise ValueError("forecast_leads must contain positive hours")
    before_hours = 6 if history_before_hours is None else int(history_before_hours)
    after_hours = 0 if history_after_hours is None else int(history_after_hours)
    if not (6 <= before_hours <= 72 and 0 <= after_hours <= 72):
        raise ValueError("history windows must satisfy 6<=before<=72 and 0<=after<=72 hours")
    request = {
        "analysis_time_utc": parsed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "era5_time_utc": aligned,
        "time_offset_minutes": delta_minutes,
        "latitude": None if latitude is None else _number(latitude, "latitude"),
        "longitude": None if longitude is None else _number(longitude, "longitude"),
        "area_north_west_south_east": area,
        "grid": "0.25/0.25",
        "surface_variables": list(variables or DEFAULT_SURFACE_VARIABLES),
        "pressure_variables": list(pressure_variables or DEFAULT_PRESSURE_VARIABLES),
        "pressure_levels_hpa": levels,
        "aurora_pressure_levels_hpa": list(DEFAULT_AURORA_LEVELS),
        "forecast_leads_hours": leads,
        "input_window_hours": 6,
        "history_before_hours": before_hours,
        "history_after_hours": after_hours,
        "data_policy": "live",
        "event_id": str(event_id or "").strip().upper(),
        "model_version": str(model_version or ""),
    }
    signature = json.dumps(request, sort_keys=True, separators=(",", ":"))
    request["request_key"] = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
    identity = request_identity(
        event_id=request.get("event_id", ""), analysis_time_utc=request["analysis_time_utc"],
        latitude=request.get("latitude"), longitude=request.get("longitude"),
        region={"area_north_west_south_east": request["area_north_west_south_east"]},
        variables=request.get("surface_variables", []) + request.get("pressure_variables", []),
        pressure_levels=request.get("pressure_levels_hpa", []),
        history_before_hours=request.get("history_before_hours"), history_after_hours=request.get("history_after_hours"),
        forecast_leads_hours=request.get("forecast_leads_hours", []), model="era5+aurora",
        model_version=request.get("model_version", ""), resolution=request.get("grid", ""),
        task="foundation_request", evidence_kind="environment",
    )
    request["canonical_request_key"] = identity["request_key"]
    root = Path(cache_root) if cache_root else Path(__file__).resolve().parents[2] / "data" / "foundation_live_cache"
    request["artifact_dir"] = str(root / (request.get("event_id") or "_live") / request["canonical_request_key"])
    request["status"] = "normalized"
    return request


def _compatible_standardized_cache(
    request: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a cached superset request that can satisfy ``request`` read-only."""
    if not manifest or manifest.get("status") != "available":
        return None
    cached = manifest.get("canonical")
    if not isinstance(cached, dict):
        return None
    if str(cached.get("event_id", "")).upper() != str(request.get("event_id", "")).upper():
        return None
    scalar_fields = ("analysis_time_utc", "era5_time_utc", "grid")
    if any(str(cached.get(key, "")) != str(request.get(key, "")) for key in scalar_fields):
        return None
    for key in ("latitude", "longitude"):
        try:
            if abs(float(cached.get(key)) - float(request.get(key))) > 1e-4:
                return None
        except (TypeError, ValueError):
            return None
    try:
        cached_area = [round(float(value), 4) for value in cached.get("area_north_west_south_east", [])]
        requested_area = [round(float(value), 4) for value in request.get("area_north_west_south_east", [])]
    except (TypeError, ValueError):
        return None
    if cached_area != requested_area:
        return None
    subset_fields = (
        "surface_variables", "pressure_variables", "pressure_levels_hpa",
        "forecast_leads_hours",
    )
    for key in subset_fields:
        if not set(request.get(key, [])).issubset(set(cached.get(key, []))):
            return None
    if int(cached.get("history_before_hours", -1)) < int(request.get("history_before_hours", 0)):
        return None
    if int(cached.get("history_after_hours", -1)) < int(request.get("history_after_hours", 0)):
        return None
    artifact_dir = Path(str(cached.get("artifact_dir", "")))
    if not artifact_dir.is_dir():
        return None
    return cached


def fetch_era5_live(request: dict[str, Any], *, artifact_dir: str | Path | None = None) -> dict[str, Any]:
    """Fetch the live request's ERA5 inputs through CDS and write a manifest."""
    if not isinstance(request, dict) or request.get("data_policy") != "live":
        raise ValueError("fetch_era5_live requires a normalized live request")
    cached_manifest = read_manifest(
        str(request.get("event_id") or "_live"),
        "standardized",
        source_role="era5_environment_analysis",
    )
    cached_superset = _compatible_standardized_cache(request, cached_manifest)
    if cached_superset is not None:
        output_dir = Path(str(cached_superset["artifact_dir"]))
        expected_key = str(cached_manifest.get("request_key", ""))
    else:
        output_dir = Path(artifact_dir or request.get("artifact_dir", ""))
        expected_key = str(request.get("canonical_request_key") or request.get("request_key") or "")
    cached_summary = output_dir / "era5_target_timeseries.json"
    cached_files = [output_dir / name for name in ("surface.nc", "pressure.nc", "static.nc")]
    readable_files = all(_netcdf_readable(path, expected_key) for path in cached_files)
    if (
        cached_manifest and cached_manifest.get("status") == "available"
        and str(cached_manifest.get("request_key", "")) == expected_key
        and cached_summary.is_file() and readable_files
    ):
        manifest_path = output_dir / "live_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else dict(request)
        manifest["cache_hit"] = True
        manifest["status"] = "ok"
        manifest["requested_request_key"] = request.get("canonical_request_key")
        manifest["canonical_request_key"] = expected_key
        manifest["cache_compatibility"] = "exact" if cached_superset is None else "cached_superset"
        manifest["target_timeseries"] = json.loads(cached_summary.read_text(encoding="utf-8"))
        return manifest
    if os.getenv("SEVIR_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            f"cache_only_miss: refusing to download ERA5 for request {expected_key}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi  # type: ignore
    except ImportError as error:
        # The HTTP service may run from the lightweight system Python while
        # the existing CasCast/Aurora environment already contains cdsapi.
        # Load only this pure-Python client from that environment instead of
        # putting its Python-3.10 binary packages on the service sys.path.
        candidates = []
        configured = os.getenv("CASCAST_PYTHON", "")
        if configured:
            candidates.append(Path(configured).resolve().parent.parent / "Lib" / "site-packages")
        candidates.append(Path(__file__).resolve().parents[3] / "CasCast" / ".venv" / "Lib" / "site-packages")
        cdsapi = None
        for candidate in candidates:
            if not (candidate / "cdsapi").exists():
                continue
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            try:
                import cdsapi as loaded_cdsapi  # type: ignore
                cdsapi = loaded_cdsapi
                break
            except ImportError:
                continue
        if cdsapi is None:
            return {
                "status": "dependency_missing",
                "error_code": "cdsapi_not_installed",
                "message": "cdsapi is required for live ERA5 retrieval",
                "request": request,
                "searched_environments": [str(item) for item in candidates],
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
    client = cdsapi.Client(timeout=60, retry_max=2)
    surface = output_dir / "surface.nc"
    pressure = output_dir / "pressure.nc"
    static = output_dir / "static.nc"
    # A CDS request can leave a truncated HDF5 file after a network timeout.
    # Existence alone is not a valid cache hit; remove only files generated by
    # this request so the normal retrieve path repairs them.
    invalid_files: set[Path] = set()
    for candidate in (surface, pressure, static):
        if candidate.exists() and not _netcdf_readable(candidate, expected_key):
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

    if not surface.exists() or surface in invalid_files:
        query = dict(common, variable=request["surface_variables"])
        surface = retrieve_checked("reanalysis-era5-single-levels", query, surface)
    if not pressure.exists() or pressure in invalid_files:
        query = dict(common, variable=request["pressure_variables"], pressure_level=[str(x) for x in request["aurora_pressure_levels_hpa"]])
        pressure = retrieve_checked("reanalysis-era5-pressure-levels", query, pressure)
    if not static.exists() or static in invalid_files:
        query = dict(common, variable=["geopotential", "land_sea_mask", "soil_type"], day=["01"], time=["00:00"])
        static = retrieve_checked("reanalysis-era5-single-levels", query, static)
    manifest = dict(request)
    manifest.update({"status": "ok", "files": {"surface": str(surface), "pressure": str(pressure), "static": str(static)}})
    summary_path = output_dir / "era5_target_timeseries.json"
    summary_script = Path(__file__).resolve().parents[2] / "scripts" / "summarize_era5_live.py"
    area = request["area_north_west_south_east"]
    target_latitude = request.get("latitude")
    target_longitude = request.get("longitude")
    if target_latitude is None:
        target_latitude = (float(area[0]) + float(area[2])) / 2.0
    if target_longitude is None:
        target_longitude = (float(area[1]) + float(area[3])) / 2.0
    stage, staged = _ascii_stage(str(request.get("canonical_request_key", request.get("request_key", "era5"))), surface, pressure)
    staged_summary = stage / summary_path.name
    summary_command = [
        _foundation_python(), str(summary_script),
        "--surface", str(staged[surface]), "--pressure", str(staged[pressure]), "--output", str(staged_summary),
        "--analysis-time", str(request["era5_time_utc"]),
        "--latitude", str(target_latitude), "--longitude", str(target_longitude),
        "--before", str(request.get("history_before_hours", 6)),
        "--after", str(request.get("history_after_hours", 0)),
        "--levels", ",".join(str(value) for value in request.get("pressure_levels_hpa", [850, 500, 250])),
    ]
    summarized = subprocess.run(summary_command, capture_output=True, text=True, timeout=300)
    if summarized.returncode == 0 and staged_summary.is_file():
        shutil.copy2(staged_summary, summary_path)
        manifest["target_timeseries"] = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest["target_timeseries_file"] = str(summary_path)
    else:
        manifest["target_timeseries_status"] = "extraction_failed"
        manifest["target_timeseries_error"] = summarized.stderr[-2000:]
    (output_dir / "live_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    event_id = str(request.get("event_id") or "_live")
    identity = {"request_key": request.get("canonical_request_key", request.get("request_key")), "canonical": request}
    write_manifest(
        event_id, "raw", identity,
        source_role="era5_environment_input",
        status="available",
        validation={"netcdf_readable": True},
        files={"surface": str(surface), "pressure": str(pressure), "static": str(static)},
        provenance={"source": "CDS ERA5", "request_key": request.get("request_key")},
    )
    standardized_status = "available" if isinstance(manifest.get("target_timeseries"), dict) else "unavailable"
    write_manifest(
        event_id, "standardized", identity,
        source_role="era5_environment_analysis",
        status=standardized_status,
        validation={"target_timeseries_available": standardized_status == "available"},
        target_timeseries_file=manifest.get("target_timeseries_file"),
        error=manifest.get("target_timeseries_error"),
        evidence=make_evidence(
            source="era5", event_id=event_id, request_key=identity["request_key"],
            time_range={"analysis_time_utc": request.get("analysis_time_utc")},
            variables=request.get("surface_variables", []) + request.get("pressure_variables", []),
            data=manifest.get("target_timeseries"),
            provenance={"source": "CDS ERA5", "files": manifest.get("files", {})},
            status=standardized_status,
        ),
    )
    return manifest


def run_aurora_live(request: dict[str, Any], *, artifact_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the checked-in Aurora wrapper on a live ERA5 artifact directory."""
    if not isinstance(request, dict) or request.get("data_policy") != "live":
        raise ValueError("run_aurora_live requires a normalized live request")
    root = Path(artifact_dir or request.get("artifact_dir", ""))
    surface, pressure, static = root / "surface.nc", root / "pressure.nc", root / "static.nc"
    if not all(path.is_file() for path in (surface, pressure, static)):
        return {"status": "inputs_missing", "error_code": "era5_live_inputs_missing", "request": request}
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_aurora_case.py"
    output = root / "aurora_forecast.npz"
    if not script.is_file():
        return {"status": "script_missing", "error_code": "aurora_runner_missing", "request": request}
    # Aurora is installed in the existing CasCast Python 3.10 environment;
    # using the service interpreter here would report a misleading missing
    # package even though the model is ready locally.
    aurora_python = _foundation_python()
    leads = [int(value) for value in request.get("forecast_leads_hours", [6])]
    # A completed request is reusable evidence. The Agent may still decide
    # how to interpret it, but must not rerun the foundation model.
    cached_summary_path = output.with_suffix(".json")
    if output.is_file() and cached_summary_path.is_file():
        try:
            cached_summary = json.loads(cached_summary_path.read_text(encoding="utf-8"))
            completed = [int(value) for value in cached_summary.get("completed_leads_hours", [])]
            if all(lead in completed for lead in leads):
                # A cache hit is deliberately read-only.  Rewriting the same
                # manifest on every question adds filesystem contention and,
                # in restricted deployments, can spend tens of seconds in
                # tempfile retries even though all evidence already exists.
                return {
                    "status": "ok",
                    "model": "AuroraPretrained",
                    "cache_hit": True,
                    "request": request,
                    "forecast": cached_summary,
                    "verification": None,
                    "verification_status": "compute_on_demand_from_forecast_and_era5",
                    "output": str(output),
                    "completed_leads_hours": completed,
                    "pending_leads_hours": [],
                    "interpretation_limit": "Aurora给出大尺度环境预测；缓存命中时不重复运行模型。",
                }
        except (OSError, ValueError, TypeError):
            pass
    if os.getenv("SEVIR_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "cache_only_miss: refusing to run Aurora for request "
            f"{request.get('canonical_request_key', request.get('request_key', ''))}"
        )
    stage, staged = _ascii_stage(str(request.get("canonical_request_key", request.get("request_key", "aurora"))), surface, pressure, static)
    staged_output = stage / output.name
    command = [
        aurora_python, _subprocess_path(script),
        "--surface", str(staged[surface]), "--pressure", str(staged[pressure]), "--static", str(staged[static]),
        "--output", str(staged_output), "--analysis-time", str(request.get("era5_time_utc", "")),
        "--latitude", str(request.get("latitude")), "--longitude", str(request.get("longitude")),
        "--leads", ",".join(str(value) for value in leads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        return {"status": "inference_failed", "error_code": "aurora_inference_failed", "stderr": completed.stderr[-4000:], "request": request}
    if staged_output.is_file():
        shutil.copy2(staged_output, output)
    staged_summary = staged_output.with_suffix(".json")
    if staged_summary.is_file():
        shutil.copy2(staged_summary, output.with_suffix(".json"))
    summary_path = output.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    requested_leads = leads
    completed_leads = [int(value) for value in summary.get("completed_leads_hours", [])]
    event_id = str(request.get("event_id") or "_live")
    identity = {"request_key": request.get("canonical_request_key", request.get("request_key")), "canonical": request}
    forecast_status = "available" if completed_leads and not [lead for lead in requested_leads if lead not in completed_leads] else "partial"
    write_manifest(
        event_id, "standardized", identity,
        source_role="aurora_environment_forecast",
        status=forecast_status,
        validation={
            "requested_leads_hours": requested_leads,
            "completed_leads_hours": completed_leads,
            "complete": forecast_status == "available",
        },
        aurora_forecast_file=str(output), completed_leads_hours=completed_leads,
        evidence=make_evidence(
            source="aurora", event_id=event_id, request_key=identity["request_key"],
            time_range={"analysis_time_utc": request.get("analysis_time_utc"), "leads_hours": completed_leads},
            variables=["temperature", "humidity", "wind", "pressure", "geopotential"],
            data=summary, provenance={"model": "AuroraPretrained", "input": str(root)},
            status=forecast_status, limitations=["not_local_vil", "not_official_warning"],
        ),
    )
    return {
        "status": "ok",
        "model": "AuroraPretrained",
        "request": request,
        "forecast": summary,
        "verification": None,
        "verification_status": "compute_on_demand_from_forecast_and_era5",
        "output": str(output),
        "completed_leads_hours": completed_leads,
        "pending_leads_hours": [lead for lead in requested_leads if lead not in completed_leads],
        "interpretation_limit": "Aurora给出大尺度环境预测；未完成的时效不能被当作已有结果，也不能替代局地VIL预报。",
    }
