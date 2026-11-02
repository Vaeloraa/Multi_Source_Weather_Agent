"""Aurora model execution and artifact-cache implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
from ..gpu import gpu_slot

from ...cache.registry import (
    CacheRegistry,
    registry_entry_key,
    versioned_artifact_identity,
)

AURORA_CACHE_NAMESPACE = "aurora.forecast.bundle.v2"
AURORA_PACKAGE_VERSION = os.getenv(
    "WEATHER_AGENT_AURORA_PACKAGE_VERSION", "2.0.1"
).strip()


def _subprocess_path(path: str | Path) -> str:
    value = str(path)
    if os.name != "nt" or value.isascii():
        return value
    try:
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        source = ctypes.c_wchar_p(value)
        size = get_short(source, None, 0)
        if size:
            buffer = ctypes.create_unicode_buffer(size + 1)
            if get_short(source, buffer, size + 1):
                return buffer.value
    except (AttributeError, OSError):
        pass
    return value


def _ascii_stage(*paths: Path) -> tuple[Path, dict[Path, Path]]:
    stage = Path(tempfile.gettempdir()) / f"weather_agent_aurora_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=True)
    mapping: dict[Path, Path] = {}
    for source in paths:
        target = stage / source.name
        if source.is_file():
            shutil.copy2(source, target)
        mapping[source] = target
    return stage, mapping


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
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
        raise ValueError("analysis_time_utc is required for Aurora inference")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cached_aurora_product(
    root: Path, request: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Find a cached forecast whose initialization and leads match the request."""

    requested_time = _parse_time(str(request.get("era5_time_utc", "")))
    requested_leads = {int(value) for value in request.get("forecast_leads_hours", [])}
    for summary_path in sorted(root.glob("aurora_forecast*.json")):
        forecast_path = summary_path.with_suffix(".npz")
        if not forecast_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            product_time = _parse_time(str(summary.get("event_input_time", "")))
            completed = {int(value) for value in summary.get("completed_leads_hours", [])}
            cached_request_key = str(summary.get("cache_request_key") or "")
            expected_request_key = str(request["request_key"])
            if not cached_request_key or cached_request_key != expected_request_key:
                continue
            expected_model_version = str(request.get("model_version") or "")
            cached_model_version = str(summary.get("model_version") or "")
            if (
                not expected_model_version
                or cached_model_version != expected_model_version
            ):
                continue
            if str(
                summary.get("aurora_package_version") or ""
            ) != AURORA_PACKAGE_VERSION:
                continue
            with np.load(forecast_path, allow_pickle=False) as payload:
                if not payload.files:
                    continue
        except (OSError, ValueError, TypeError):
            continue
        if product_time == requested_time and requested_leads.issubset(completed):
            summary = dict(summary)
            summary["cache_identity_status"] = "versioned_bundle"
            identity = {
                "request_key": expected_request_key,
                "canonical": request["cache_parameters"],
            }
            registry = CacheRegistry()
            if registry.resolve(AURORA_CACHE_NAMESPACE, identity) is None:
                registry.register(
                    AURORA_CACHE_NAMESPACE,
                    identity,
                    forecast_path,
                    metadata={
                        "model": summary.get("model"),
                        "model_version": summary.get("model_version"),
                        "completed_leads_hours": sorted(completed),
                        "summary_path": str(summary_path),
                    },
                )
            return forecast_path, summary
    return None


def _validate_fresh_aurora_product(
    forecast_path: Path, summary_path: Path, request: dict[str, Any]
) -> dict[str, Any]:
    """Validate the complete versioned worker bundle before publication."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != "ok":
        raise ValueError("Aurora summary does not report status=ok")
    expected_key = str(request["request_key"])
    if not expected_key or str(summary.get("cache_request_key") or "") != expected_key:
        raise ValueError("Aurora output cache identity does not match request")
    expected_model = str(request.get("model_version") or "")
    if not expected_model or str(summary.get("model_version") or "") != expected_model:
        raise ValueError("Aurora output model version does not match request")
    if str(summary.get("aurora_package_version") or "") != AURORA_PACKAGE_VERSION:
        raise ValueError("Aurora package version does not match configured runtime")
    product_time = _parse_time(str(summary.get("event_input_time") or ""))
    if product_time != _parse_time(str(request.get("era5_time_utc") or "")):
        raise ValueError("Aurora output initialization time does not match request")
    requested_leads = {
        int(value) for value in request.get("forecast_leads_hours", [])
    }
    completed_leads = {
        int(value) for value in summary.get("completed_leads_hours", [])
    }
    if not requested_leads or not requested_leads.issubset(completed_leads):
        raise ValueError("Aurora output does not contain every requested lead")
    with np.load(forecast_path, allow_pickle=False) as payload:
        required = {f"lead_{lead:03d}_2t" for lead in requested_leads}
        if not required.issubset(set(payload.files)):
            raise ValueError("Aurora NPZ is missing requested lead arrays")
        # Materialize one array at a time so truncated compressed members are
        # detected before publication without retaining the full bundle.
        for name in payload.files:
            if np.asarray(payload[name]).size == 0:
                raise ValueError(f"Aurora NPZ member is empty: {name}")
    summary["cache_identity_status"] = "versioned_bundle"
    return summary


def run_aurora_live(
    request: dict[str, Any],
    *,
    artifact_dir: str | Path | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Run Aurora on persisted ERA5/ERA5T reanalysis inputs.

    Live means on-demand execution, not real-time ERA5 data.
    """
    if not isinstance(request, dict) or request.get("data_policy") != "live":
        raise ValueError("run_aurora_live requires a normalized live request")
    root = Path(artifact_dir or request.get("artifact_dir", ""))
    surface, pressure, static = root / "surface.nc", root / "pressure.nc", root / "static.nc"
    if not all(path.is_file() for path in (surface, pressure, static)):
        return {"status": "inputs_missing", "error_code": "era5_live_inputs_missing", "request": request}
    output = root / "aurora_forecast.npz"
    leads = [int(value) for value in request.get("forecast_leads_hours", [6])]
    # A completed request is reusable evidence. The Agent may still decide
    # how to interpret it, but must not rerun the foundation model.
    cached_product = _cached_aurora_product(root, request)
    if cached_product is not None:
        output, cached_summary = cached_product
        completed = [int(value) for value in cached_summary.get("completed_leads_hours", [])]
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
            "interpretation_limit": "Aurora provides large-scale forecasts. Matching cached products are reused.",
        }
    if cache_only or os.getenv("WEATHER_AGENT_CACHE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "cache_only_miss: refusing to run Aurora for request "
            f"{request['request_key']}"
        )
    # Production workers belong to the package; cleaning one-off scripts must
    # not silently break the service's cache-miss execution path.
    script = Path(__file__).with_name("worker.py")
    if not script.is_file():
        return {"status": "script_missing", "error_code": "aurora_runner_missing", "request": request}
    # Aurora is installed in the existing CasCast Python 3.10 environment;
    # using the service interpreter here would report a misleading missing
    # package even though the model is ready locally.
    aurora_python = _foundation_python()
    lease_identity = versioned_artifact_identity(
        "cache.build.lease",
        source={"namespace": "aurora.forecast", "artifact": str(output.resolve())},
    )
    lease_key = registry_entry_key("cache.build.lease", lease_identity)
    lease_owner = uuid.uuid4().hex
    registry = CacheRegistry()
    if not registry.acquire_lease(
        lease_key, lease_owner, lease_seconds=7800
    ):
        return {
            "status": "in_progress",
            "error_code": "aurora_cache_build_in_progress",
            "request": request,
        }
    stage: Path | None = None
    published_summary: dict[str, Any] | None = None
    try:
        # Close the race between the first lookup and lease acquisition.
        cached_product = _cached_aurora_product(root, request)
        if cached_product is not None:
            cached_output, cached_summary = cached_product
            completed_leads = [
                int(value) for value in cached_summary.get("completed_leads_hours", [])
            ]
            return {
                "status": "ok",
                "model": "AuroraPretrained",
                "cache_hit": True,
                "request": request,
                "forecast": cached_summary,
                "verification": None,
                "verification_status": "compute_on_demand_from_forecast_and_era5",
                "output": str(cached_output),
                "completed_leads_hours": completed_leads,
                "pending_leads_hours": [],
                "interpretation_limit": "Aurora provides large-scale forecasts. Matching cached products are reused.",
            }
        stage, staged = _ascii_stage(surface, pressure, static)
        staged_output = stage / output.name
        command = [
            aurora_python, _subprocess_path(script),
            "--surface", str(staged[surface]), "--pressure", str(staged[pressure]), "--static", str(staged[static]),
            "--output", str(staged_output), "--analysis-time", str(request.get("era5_time_utc", "")),
            "--leads", ",".join(str(value) for value in leads),
            "--cache-request-key", str(request["request_key"]),
            "--model-version", str(request.get("model_version") or ""),
        ]
        foundation_environment = os.environ.copy()
        foundation_environment.pop("PYTHONPATH", None)
        for variable in (
            "TEMP", "TMP", "TMPDIR", "WANDB_DIR", "WANDB_CACHE_DIR",
            "WANDB_CONFIG_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR",
        ):
            foundation_environment[variable] = str(stage)
        try:
            with gpu_slot(timeout=1800):
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=foundation_environment,
                )
        except subprocess.TimeoutExpired:
            return {
                "status": "inference_failed",
                "error_code": "aurora_inference_timeout",
                "request": request,
            }
        except OSError as error:
            return {
                "status": "inference_failed",
                "error_code": "aurora_worker_start_failed",
                "stderr": str(error)[-4000:],
                "request": request,
            }
        if completed.returncode != 0:
            return {"status": "inference_failed", "error_code": "aurora_inference_failed", "stderr": completed.stderr[-4000:], "request": request}
        staged_summary = staged_output.with_suffix(".json")
        if not staged_output.is_file() or not staged_summary.is_file():
            return {
                "status": "inference_failed",
                "error_code": "aurora_output_incomplete",
                "stderr": (completed.stderr or completed.stdout or "")[-4000:],
                "request": request,
            }
        try:
            published_summary = _validate_fresh_aurora_product(
                staged_output, staged_summary, request
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            return {
                "status": "inference_failed",
                "error_code": "aurora_output_validation_failed",
                "stderr": str(error)[-4000:],
                "request": request,
            }
        # The request-key directory is immutable for compatible output. Each
        # file is copied to a sibling temporary path before atomic replace;
        # the summary is published last and therefore acts as the read gate.
        _atomic_copy(staged_output, output)
        _atomic_copy(staged_summary, output.with_suffix(".json"))
    finally:
        registry.release_lease(lease_key, lease_owner)
        if stage is not None:
            try:
                shutil.rmtree(stage)
            except OSError:
                pass
    summary_path = output.with_suffix(".json")
    summary = published_summary or {}
    if summary:
        CacheRegistry().register(
            AURORA_CACHE_NAMESPACE,
            {
                "request_key": str(request["request_key"]),
                "canonical": request["cache_parameters"],
            },
            output,
            metadata={
                "model": summary.get("model"),
                "model_version": summary.get("model_version"),
                "completed_leads_hours": summary.get("completed_leads_hours", []),
                "summary_path": str(summary_path),
            },
        )
    completed_leads = [int(value) for value in summary.get("completed_leads_hours", [])]
    return {
        "status": "ok",
        "model": "AuroraPretrained",
        "request": request,
        "forecast": summary,
        "verification": None,
        "verification_status": "compute_on_demand_from_forecast_and_era5",
        "output": str(output),
        "completed_leads_hours": completed_leads,
        "pending_leads_hours": [lead for lead in leads if lead not in completed_leads],
        "interpretation_limit": "Aurora provides large-scale forecasts. Only completed lead times are available; local VIL forecasts are a separate product.",
    }
