"""Generic request-scoped Evidence Ledger."""
from __future__ import annotations

from typing import Any
import hashlib
import json


def evidence_key(api: str, arguments: dict[str, Any]) -> str:
    raw = json.dumps({"api": api, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


class EvidenceLedger:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def record(self, api: str, arguments: dict[str, Any], result: Any = None, status: str = "ok") -> dict[str, Any]:
        key = evidence_key(api, arguments)
        entry = {"key": key, "api": api, "arguments": dict(arguments), "status": status}
        if status == "ok" and isinstance(result, dict):
            entry["schema_version"] = result.get("schema_version")
            entry["evidence_id"] = result.get("forecast_evidence_id") or result.get("artifact_id")
        self.entries[key] = entry
        return entry

    def completed(self) -> list[dict[str, Any]]:
        return [item for item in self.entries.values() if item.get("status") == "ok"]

    def prompt_context(self) -> dict[str, Any]:
        return {"completed_evidence": self.completed(), "reuse_rule": "reuse completed evidence unless arguments materially change"}
