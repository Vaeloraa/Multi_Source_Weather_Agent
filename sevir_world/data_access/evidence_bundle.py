"""Model-neutral evidence bundles and coverage checks."""
from __future__ import annotations

from typing import Any, Iterable


ROLES = {
    "sevir": "local_observation",
    "cascast": "local_forecast",
    "era5": "environment_reanalysis",
    "aurora": "environment_forecast",
}


def make_evidence(
    *, source: str, event_id: str = "", request_key: str = "", time_range: dict[str, Any] | None = None,
    variables: Iterable[str] = (), data: Any = None, provenance: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None, limitations: Iterable[str] = (),
    status: str = "available", role: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source, "role": role or ROLES.get(source, "unknown"),
        "event_id": event_id, "request_key": request_key,
        "time_range": time_range or {}, "variables": sorted({str(x) for x in variables}),
        "data": data, "provenance": provenance or {}, "validation": validation or {},
        "limitations": list(limitations), "status": status,
    }


def bundle(evidence: Iterable[dict[str, Any]], *, event_id: str = "", request_key: str = "") -> dict[str, Any]:
    items = [item for item in evidence if isinstance(item, dict)]
    by_source = {str(item.get("source")): item for item in items if item.get("source")}
    return {
        "event_id": event_id,
        "request_key": request_key,
        "evidence": by_source,
        "available_sources": sorted(by_source),
        "missing_sources": sorted(set(ROLES) - set(by_source)),
        "status": "available" if items and all(item.get("status") == "available" for item in items) else "partial",
    }


def required_sources(requirements: Any) -> set[str]:
    if isinstance(requirements, dict) and requirements.get("required_evidence"):
        return {str(item.get("source")) for item in requirements["required_evidence"] if isinstance(item, dict) and item.get("source")}
    required: set[str] = set()
    if getattr(requirements, "needs_local_observation", False):
        required.add("sevir")
    if getattr(requirements, "needs_local_forecast", False):
        required.add("cascast")
    if getattr(requirements, "needs_environment_history", False):
        required.add("era5")
    if getattr(requirements, "scale", "") == "large_scale" and getattr(requirements, "task", "") in {"forecast", "verification"}:
        required.add("aurora")
    if getattr(requirements, "needs_verification", False):
        if "cascast" in required:
            required.add("sevir")
        if "aurora" in required:
            required.add("era5")
    return required


def check_coverage(requirements: Any, evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    required = required_sources(requirements)
    available = set(evidence_bundle.get("available_sources", []))
    missing = sorted(required - available)
    return {
        "status": "ok" if not missing else "evidence_insufficient",
        "required_sources": sorted(required),
        "available_sources": sorted(available),
        "missing_sources": missing,
    }


def check_declared_coverage(required_evidence: Iterable[dict[str, Any]], evidence_bundle: dict[str, Any]) -> dict[str, Any]:
    """Coverage gate for the serialized request contract used by the API."""
    required = {str(item.get("source")) for item in required_evidence if isinstance(item, dict) and item.get("source")}
    available = set(evidence_bundle.get("available_sources", []))
    missing = sorted(required - available)
    return {"status": "ok" if not missing else "evidence_insufficient", "required_sources": sorted(required), "available_sources": sorted(available), "missing_sources": missing}


def can_fallback(requirements: Any, evidence_bundle: dict[str, Any]) -> bool:
    return check_coverage(requirements, evidence_bundle)["status"] == "ok"
