"""Event-level cache index for ERA5/Aurora evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .cache_layers import layer_status
from .foundation import get_case_foundation_artifacts


def foundation_cache_status(event_id: str) -> dict[str, Any]:
    root = ROOT / "data" / "era5_cases" / str(event_id) / "artifacts"
    evidence = root / "case_evidence.json"
    files = {p.name: int(p.stat().st_size) for p in root.iterdir() if p.is_file()} if root.is_dir() else {}
    registry = get_case_foundation_artifacts(str(event_id))
    registry_hit = registry.get("status") == "ok"
    payload: dict[str, Any] = {
        "event_id": str(event_id),
        "cache_root": str(root),
        "evidence_available": evidence.is_file() or bool(registry.get("evidence")),
        "files": files or {
            str(item.get("relative_path") or item.get("name")): int(item.get("size_bytes", 0))
            for item in registry.get("artifacts", [])
            if item.get("available")
        },
        "cache_hit": evidence.is_file() or registry_hit,
        "source": registry.get("cache_source", "local_event_cache"),
    }
    # The four-layer cache is additive to the legacy case_evidence layout.
    # Keeping both fields makes old clients compatible while exposing the
    # request-key-aware cache status to new adapters.
    try:
        payload["layers"] = layer_status(str(event_id))
    except (OSError, ValueError):
        payload["layers"] = {"event_id": str(event_id), "cache_hit": False, "layers": {}}
    if evidence.is_file():
        try:
            data = json.loads(evidence.read_text(encoding="utf-8"))
            aurora = data.get("aurora", {}) if isinstance(data, dict) else {}
            payload["aurora_leads"] = aurora.get("rollout_leads_hours", [])
            payload["verification_status"] = aurora.get("verification_status")
        except (OSError, ValueError):
            payload["cache_hit"] = False
            payload["error"] = "evidence_json_invalid"
    elif registry_hit:
        aurora = (registry.get("evidence") or {}).get("aurora", {})
        payload["aurora_leads"] = aurora.get("rollout_leads_hours", [])
        payload["verification_status"] = aurora.get("verification_status")
    return payload
