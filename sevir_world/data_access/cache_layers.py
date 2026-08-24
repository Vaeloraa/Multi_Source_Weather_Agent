"""Four-layer evidence cache with atomic JSON metadata writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import ROOT


LAYER_NAMES = ("raw", "standardized", "verification", "presentation")
MANIFEST_SCHEMA_VERSION = "evidence-cache-v1"
_ANSWER_FIELDS = frozenset({"answer", "final_answer", "natural_language_answer", "response_text"})


def cache_root(event_id: str, layer: str) -> Path:
    if layer not in LAYER_NAMES:
        raise ValueError(f"unknown cache layer: {layer}")
    return ROOT / "data" / "foundation_cache" / str(event_id).strip().upper() / layer


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _contains_answer_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).strip().lower() in _ANSWER_FIELDS or _contains_answer_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_answer_field(item) for item in value)
    return False


def _legacy_role(layer: str, payload: dict[str, Any]) -> str:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    source = str(evidence.get("source") or payload.get("model") or "").lower()
    if layer == "raw":
        return "era5_environment_input"
    if layer == "verification":
        return "model_forecast_verification"
    if layer == "presentation":
        return "presentation_artifact"
    if "aurora" in source or payload.get("aurora_forecast_file"):
        return "aurora_environment_forecast"
    return "era5_environment_analysis"


def _read_container(event_id: str, layer: str) -> dict[str, Any] | None:
    path = cache_root(event_id, layer) / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") == MANIFEST_SCHEMA_VERSION and isinstance(payload.get("entries"), dict):
        return payload
    role = _legacy_role(layer, payload)
    entry = dict(payload)
    entry.update({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_role": role,
        "evidence_role": "presentation_only" if layer == "presentation" else "scientific_evidence",
        "contains_natural_language_answer": False,
        "validation": dict(payload.get("validation") or {}),
    })
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "event_id": str(event_id).strip().upper(),
        "layer": layer,
        "entries": {role: entry},
    }


def write_manifest(
    event_id: str,
    layer: str,
    identity: dict[str, Any],
    *,
    source_role: str,
    status: str = "available",
    validation: dict[str, Any] | None = None,
    **metadata: Any,
) -> Path:
    role = str(source_role).strip()
    if not role:
        raise ValueError("source_role is required for cache evidence")
    if _contains_answer_field(metadata):
        raise ValueError("natural-language answers cannot be stored as cache evidence")
    event = str(event_id).strip().upper()
    existing = _read_container(event, layer) or {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "event_id": event,
        "layer": layer,
        "entries": {},
    }
    entry = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "event_id": event,
        "layer": layer,
        "source_role": role,
        "evidence_role": "presentation_only" if layer == "presentation" else "scientific_evidence",
        "contains_natural_language_answer": False,
        "request_key": identity.get("request_key"),
        "canonical": identity.get("canonical", {}),
        "status": status,
        "validation": dict(validation or {}),
        **metadata,
    }
    existing["entries"][role] = entry
    payload = existing
    path = cache_root(event_id, layer) / "manifest.json"
    _atomic_json(path, payload)
    return path


def read_manifest(
    event_id: str, layer: str, *, source_role: str | None = None
) -> dict[str, Any] | None:
    container = _read_container(event_id, layer)
    if container is None:
        return None
    entries = container["entries"]
    if source_role is not None:
        value = entries.get(str(source_role))
        return dict(value) if isinstance(value, dict) else None
    if len(entries) == 1:
        return dict(next(iter(entries.values())))
    return container


def layer_status(event_id: str, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"event_id": event_id, "layers": {}, "cache_hit": False}
    for layer in LAYER_NAMES:
        container = _read_container(event_id, layer)
        entries = list((container or {}).get("entries", {}).values())
        compatible_entries = [
            item for item in entries
            if item.get("status") == "available"
            and (not identity or item.get("request_key") == identity.get("request_key"))
        ]
        compatible = bool(compatible_entries)
        result["layers"][layer] = {
            "available": bool(entries),
            "compatible": compatible,
            "source_roles": sorted(str(item.get("source_role")) for item in entries),
            "compatible_source_roles": sorted(
                str(item.get("source_role")) for item in compatible_entries
            ),
        }
        result["cache_hit"] = result["cache_hit"] or compatible
    return result


def resolve_cached_layer(event_id: str, identity: dict[str, Any], *, source_role: str | None = None, preferred: tuple[str, ...] = ("presentation", "standardized", "verification", "raw")) -> dict[str, Any]:
    """Return the first compatible cache layer and its manifest.

    This is intentionally read-only.  Adapters call it before downloading or
    inferring, so a repeated question can reuse an existing artifact without
    knowing the storage layout.
    """
    for layer in preferred:
        manifest = read_manifest(event_id, layer, source_role=source_role)
        if manifest and manifest.get("status") == "available" and manifest.get("request_key") == identity.get("request_key"):
            return {"cache_hit": True, "layer": layer, "manifest": manifest}
    return {"cache_hit": False, "layer": None, "manifest": None}


def write_verification(event_id: str, identity: dict[str, Any], *, model: str, truth_source: str, metrics: dict[str, Any], artifacts: list[str] | None = None) -> Path:
    """Persist model-vs-truth metrics as a separate verification layer."""
    return write_manifest(
        event_id, "verification", identity,
        source_role="model_forecast_verification",
        status="available",
        validation={"truth_source": truth_source, "verified": True},
        model=model, truth_source=truth_source, metrics=metrics, artifacts=artifacts or [],
    )
