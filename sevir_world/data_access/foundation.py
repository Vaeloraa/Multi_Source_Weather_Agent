"""Capability cards and case-level artifacts for external foundation models.

This module deliberately keeps model outputs separate.  It exposes metadata and
bounded artifact references to the Agent; numerical ERA5/Aurora arrays remain
on disk and are never silently mixed with SEVIR VIL values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import ROOT
from .capabilities import capability_snapshot
from .tool_contracts import tool_card_snapshot


ERA5_CASE_ROOT = ROOT / "data" / "era5_cases"
LIVE_CASE_ROOT = ROOT / "data" / "foundation_live_cache"


def _safe_event_id(event_id: str) -> str:
    value = str(event_id or "").strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("event_id must contain only letters, digits, '.', '_' or '-'.")
    return value


def get_model_capabilities() -> dict[str, Any]:
    """Return capability cards used for model selection and evidence boundaries."""
    # The registry in capabilities.py is the single source of truth.  The
    # product-facing fields below are a compatibility projection for old UI
    # clients; they are never used for routing.
    registry = capability_snapshot()
    legacy_specs = {
    "sevir": {"role": "historical_observation", "input_modalities": ["SEVIR VIL"], "spatial_scale": "local 384x384 grid", "temporal_scale": "5-minute frames", "forecast_horizon": None, "verification": "later SEVIR observations", "can_support": ["observed local VIL evolution"], "cannot_support": ["large-scale environmental forecast"], "task_types": ["observation"]},
    "cascast": {"role": "local_forecast", "input_modalities": ["13 SEVIR VIL frames"], "spatial_scale": "local SEVIR grid", "temporal_scale": "5-minute leads", "forecast_horizon": "5-60 minutes", "verification": "held-out future SEVIR VIL", "can_support": ["local VIL short-term forecast"], "cannot_support": ["synoptic environmental explanation by itself"], "task_types": ["local_forecast"]},
    "era5": {"role": "environmental_analysis", "input_modalities": ["reanalysis fields"], "spatial_scale": "regional 0.25-degree grid", "temporal_scale": "hourly", "forecast_horizon": None, "verification": "reanalysis provenance and consistency checks", "can_support": ["event-time environmental context", "regional environmental time series"], "cannot_support": ["direct local VIL prediction"], "task_types": ["historical_analysis", "historical_timeseries"]},
    "aurora": {"role": "large_scale_forecast", "input_modalities": ["ERA5 0.25-degree fields"], "spatial_scale": "global/regional 0.25-degree grid", "temporal_scale": "model forecast time steps", "forecast_horizon": "configured by checkpoint", "verification": "subsequent ERA5 fields", "can_support": ["future regional environmental tendency"], "cannot_support": ["direct local VIL or storm-cell certainty"], "task_types": ["environment_forecast"]},
    }
    cards = []
    for model, card in registry.items():
        cards.append({"model": model, **legacy_specs.get(model, {}), "outputs": card.get("outputs", []), "registry": card})
    return {
        "status": "ok",
        "task": "model_capabilities",
        "cards": cards,
        "tool_cards": tool_card_snapshot(),
        "interpretation_limit": "Cards describe intended evidence roles; they are not accuracy guarantees.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _live_manifest_projection(manifest: dict[str, Any], case_root: Path) -> dict[str, Any]:
    """Keep request identity/provenance, not embedded derived answers."""

    keys = (
        "status", "event_id", "analysis_time_utc", "era5_time_utc",
        "time_offset_minutes", "latitude", "longitude",
        "area_north_west_south_east", "grid", "forecast_leads_hours",
        "input_window_hours", "history_before_hours", "history_after_hours",
        "request_key",
    )
    value = {key: manifest.get(key) for key in keys if key in manifest}
    value["cache_dir"] = str(case_root)
    value["cache_kind"] = "foundation_live_cache"
    return value


def _live_evidence(case_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    target = _read_json(case_root / "era5_target_timeseries.json")
    aurora_summary = _read_json(case_root / "aurora_forecast.json")
    verification = _read_json(
        case_root / "verification" / "aurora_vs_era5_verification.json"
    )
    verification_names = [
        path.name
        for path in (
            case_root / "verification" / "aurora_vs_era5_verification.json",
            case_root / "verification" / "aurora_vs_era5_timeseries.png",
            case_root / "verification" / "aurora_error_maps.png",
        )
        if path.is_file()
    ]
    return {
        "era5": {
            "status": target.get("status", manifest.get("status")),
            "role": "retrospective_environment",
            "analysis_time_utc": target.get("analysis_time_utc", manifest.get("era5_time_utc")),
            "target_timeseries": {
                "requested_target": target.get("requested_target"),
                "selected_target": target.get("selected_target"),
                "coverage": target.get("coverage"),
                "endpoint_change": target.get("endpoint_change"),
            },
        },
        "aurora": {
            "status": aurora_summary.get("status", manifest.get("status")),
            "role": "large_scale_environment_forecast",
            "model": aurora_summary.get("model", "Aurora"),
            "rollout_leads_hours": verification.get(
                "lead_hours", aurora_summary.get("completed_leads_hours", [])
            ),
            "rollout_target_point_forecast": verification.get("target_point_forecast"),
            "rollout_target_point_era5": verification.get("target_point_era5"),
            "rollout_surface_region_mae": verification.get("surface_region_metrics"),
            "verification_status": verification.get("verification_status"),
            "verification_artifacts": verification_names,
        },
    }


def _live_case_foundation_artifacts(event_id: str) -> dict[str, Any] | None:
    event_root = LIVE_CASE_ROOT / event_id
    if not event_root.is_dir():
        return None
    manifests = sorted(
        event_root.glob("*/live_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        manifest_raw = _read_json(manifest_path)
        if not manifest_raw:
            continue
        case_root = manifest_path.parent
        items: list[dict[str, Any]] = []
        for path in sorted(case_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".json", ".nc", ".npz"}:
                continue
            name = path.name.lower()
            if "aurora" in name:
                model = "aurora"
                role = "verification" if "verification" in name or path.parent.name == "verification" else "forecast"
            elif path.suffix.lower() == ".nc" or "era5" in name:
                model, role = "era5", "analysis"
            else:
                model, role = "foundation", "manifest"
            items.append({
                "name": path.name,
                "relative_path": path.relative_to(case_root).as_posix(),
                "path": str(path),
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": int(path.stat().st_size),
                "model": model,
                "role": role,
                "available": True,
            })
        return {
            "status": "ok",
            "task": "case_foundation_artifacts",
            "event_id": event_id,
            "manifest": _live_manifest_projection(manifest_raw, case_root),
            "evidence": _live_evidence(case_root, manifest_raw),
            "artifacts": items,
            "available_count": len(items),
            "cache_source": "foundation_live_cache",
            "interpretation_limit": "Artifact availability is provenance, not an accuracy guarantee; inspect verification evidence separately.",
        }
    return None


def get_case_foundation_artifacts(event_id: str, include_missing: bool = False) -> dict[str, Any]:
    """List available ERA5/Aurora artifacts for one existing SEVIR case."""
    safe_id = _safe_event_id(event_id)
    case_root = ERA5_CASE_ROOT / safe_id
    manifest_path = case_root / "manifest.json"
    if not manifest_path.is_file():
        live_case = _live_case_foundation_artifacts(safe_id)
        if live_case is not None:
            return live_case
        return {
            "status": "not_found",
            "task": "case_foundation_artifacts",
            "event_id": safe_id,
            "reason": f"No foundation-model manifest found at {manifest_path}.",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_root = case_root / "artifacts"
    items: list[dict[str, Any]] = []
    for path in sorted(artifact_root.glob("*")) if artifact_root.is_dir() else []:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".png", ".json", ".nc", ".npz"}:
            continue
        name = path.name.lower()
        if "aurora" in name:
            model, role = "aurora", "forecast"
        elif suffix == ".nc" and "era5" in name or "surface" in name or "pressure" in name:
            model, role = "era5", "analysis"
        else:
            model, role = "era5", "derived_artifact"
        items.append({
            "name": path.name,
            "path": str(path),
            "format": suffix.lstrip("."),
            "model": model,
            "role": role,
            "preferred": path.name.startswith("aurora_forecast_event."),
            "experimental": path.name in {"aurora_forecast.npz", "aurora_forecast.json", "aurora_forecast_aligned.npz", "aurora_forecast_aligned.json"},
            "available": True,
        })
    if include_missing:
        for name, model, role in (
            ("aurora_forecast.nc", "aurora", "forecast"),
            ("aurora_vs_era5_verification.json", "aurora", "verification"),
        ):
            if not any(item["name"] == name for item in items):
                items.append({"name": name, "path": str(artifact_root / name), "format": name.rsplit(".", 1)[-1], "model": model, "role": role, "available": False})
    return {
        "status": "ok",
        "task": "case_foundation_artifacts",
        "event_id": safe_id,
        "manifest": manifest,
        "evidence": _read_json(artifact_root / "case_evidence.json"),
        "artifacts": items,
        "available_count": sum(bool(item["available"]) for item in items),
        "cache_source": "legacy_era5_case_cache",
        "interpretation_limit": "Artifact availability does not by itself establish scientific validity; use the model-specific verification record.",
    }
