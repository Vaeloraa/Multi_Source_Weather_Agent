"""Deterministic validation for ERA5/Aurora case evidence."""
from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_foundation_evidence(evidence: Any) -> dict[str, Any]:
    """Return a strict, JSON-safe validation report without filling values."""
    report: dict[str, Any] = {"status": "ok", "errors": [], "warnings": []}
    if not isinstance(evidence, dict):
        return {"status": "error", "errors": ["evidence_missing"], "warnings": []}
    for source in ("era5", "aurora"):
        item = evidence.get(source)
        if isinstance(item, dict) and item:
            if item.get("status") in {"unavailable", "error", "missing"}:
                report["errors"].append(f"{source}_status_{item.get('status')}")
            if "forecast" in item and "verification" in item:
                report["warnings"].append(f"{source}_forecast_and_verification_present_in_one_legacy_section")
            if item.get("data") is None and not any(key in item for key in ("target_timeseries", "target_point", "rollout_target_point_forecast")):
                report["warnings"].append(f"{source}_structured_data_missing")
    for source in ("era5", "aurora"):
        if not isinstance(evidence.get(source), dict):
            report["errors"].append(f"{source}_evidence_missing")
    aurora = evidence.get("aurora") if isinstance(evidence.get("aurora"), dict) else {}
    leads = aurora.get("rollout_leads_hours")
    if leads != [6, 12, 18, 24]:
        report["errors"].append("aurora_rollout_leads_incomplete")
    if aurora.get("verification_status") != "full_domain_multi_lead":
        report["warnings"].append("aurora_verification_not_full_domain_multi_lead")
    # Zero is a valid wind value, but zero temperature/pressure/humidity in a
    # populated case is almost certainly a failed serialization or placeholder.
    for section, fields in {
        "era5": ("target_2m_temperature_c", "target_mean_sea_level_pressure_hpa", "target_850hpa_specific_humidity_gkg"),
        "aurora": (),
    }.items():
        payload = evidence.get(section) if isinstance(evidence.get(section), dict) else {}
        for field in fields:
            if field in payload and not _finite(payload[field]):
                report["errors"].append(f"{section}.{field}_not_numeric")
            if field in payload and float(payload[field]) == 0.0:
                report["errors"].append(f"{section}.{field}_zero_placeholder")
    if report["errors"]:
        report["status"] = "error"
    return report


def build_foundation_bundle(event_id: str, evidence: Any, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return one stable envelope while retaining raw evidence compatibility."""
    validation = validate_foundation_evidence(evidence)
    available = {str(item.get("name")) for item in (artifacts or []) if item.get("available")}
    aurora = evidence.get("aurora", {}) if isinstance(evidence, dict) else {}
    required = set(aurora.get("verification_artifacts", [])) if isinstance(aurora, dict) else set()
    missing = sorted(required - available) if available else []
    if missing:
        validation["errors"].append("missing_verification_artifacts:" + ",".join(missing))
        validation["status"] = "error"
    raw = evidence if isinstance(evidence, dict) else {}
    aurora = raw.get("aurora", {}) if isinstance(raw.get("aurora"), dict) else {}
    standardized = {
        "sevir": raw.get("sevir"),
        "cascast": raw.get("cascast"),
        "era5": raw.get("era5"),
        "aurora": {
            "role": aurora.get("role"),
            "model": aurora.get("model"),
            "forecast": {
                "lead_hours": aurora.get("rollout_leads_hours"),
                "target_point": aurora.get("rollout_target_point_forecast"),
            },
            "verification": {
                "status": aurora.get("verification_status"),
                "truth_source": "subsequent_era5",
                "target_point_truth": aurora.get("rollout_target_point_era5"),
                "regional_mae": aurora.get("rollout_surface_region_mae"),
                "artifacts": aurora.get("verification_artifacts", []),
            },
            "supported_conclusion": aurora.get("supported_conclusion"),
            "limit": aurora.get("limit"),
        },
    }
    return {
        "status": "ok" if validation["status"] == "ok" else "invalid",
        "event_id": event_id,
        "evidence": standardized,
        "raw_evidence": evidence if isinstance(evidence, dict) else None,
        "artifacts": artifacts or [],
        "validation": validation,
    }
