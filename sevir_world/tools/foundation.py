from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent.schema import contractize
from ..data_access.config import ROOT
from ..data_access.foundation import get_case_foundation_artifacts, get_model_capabilities
from ..data_access.foundation_validation import build_foundation_bundle
from ..data_access.foundation_cache import foundation_cache_status
from .live_foundation import (
    era5_snapshot,
    era5_timeseries,
    fetch_era5,
    forecast_aurora,
    live_preflight,
    region_fields,
)
from .base import WorldComponent, resolve_event_id


class FoundationModelContext(WorldComponent):
    """Load case-level ERA5/Aurora evidence without mixing it with VIL arrays."""

    @staticmethod
    def _adapt(result: dict[str, Any], role: str, capabilities: list[str]) -> dict[str, Any]:
        """Expose one stable foundation envelope while keeping legacy keys."""
        adapted = dict(result)
        adapted.setdefault("role", role)
        adapted.setdefault(
            "capabilities",
            capabilities if adapted.get("status") in {"ok", "planned", "normalized"} else [],
        )
        request = adapted.get("request")
        adapted.setdefault("provenance", {
            "source": "foundation_adapter",
            "request_key": (
                request.get("request_key") if isinstance(request, dict)
                else adapted.get("request_key")
            ),
        })
        if "data" not in adapted:
            if isinstance(adapted.get("target_timeseries"), dict):
                adapted["data"] = adapted["target_timeseries"]
            elif isinstance(adapted.get("forecast"), dict):
                adapted["data"] = adapted["forecast"]
            elif isinstance(adapted.get("evidence_bundle"), dict):
                adapted["data"] = adapted["evidence_bundle"]
            elif role == "live_request":
                adapted["data"] = {
                    key: value for key, value in adapted.items()
                    if key not in {"data", "provenance", "limitations"}
                }
            else:
                adapted["data"] = {}
        adapted.setdefault(
            "limitations",
            [adapted.get("interpretation_limit")] if adapted.get("interpretation_limit") else [],
        )
        # `_call` already exposes ContractDict, and Agent programs rely on its
        # stable `result.field` access. Re-contractize after adding adapter
        # metadata instead of leaking a plain top-level dict from live methods.
        return contractize(adapted)

    def capabilities(self) -> dict[str, Any]:
        """Return SEVIR, CasCast, ERA5 and Aurora capability/evidence boundaries."""
        return self._call("foundation.capabilities", {}, get_model_capabilities)

    def case(self, event_id: Any) -> dict[str, Any]:
        """Return registered environmental-model artifacts and compact case evidence."""
        resolved = resolve_event_id(event_id)

        def load_case(event_id: str) -> dict[str, Any]:
            registry = get_case_foundation_artifacts(event_id)
            evidence_path = ROOT / "data" / "era5_cases" / event_id / "artifacts" / "case_evidence.json"
            evidence = None
            if evidence_path.is_file():
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            else:
                evidence = registry.get("evidence")
            bundle = build_foundation_bundle(event_id, evidence, registry.get("artifacts", []))
            return self._adapt({
                "status": bundle["status"] if registry.get("status") == "ok" else registry.get("status", "not_found"),
                "task": "foundation_case",
                "event_id": event_id,
                # Keep the explicit short key expected by the Agent contract;
                # case_evidence remains as the descriptive alias for clients.
                "evidence": evidence,
                "case_evidence": evidence,
                "evidence_bundle": bundle,
                "cache": foundation_cache_status(event_id),
                "validation": bundle["validation"],
                "artifacts": registry.get("artifacts", []),
                "manifest": registry.get("manifest"),
                "interpretation_limit": (
                    "SEVIR/CasCast describe local encoded VIL; ERA5 describes retrospective "
                    "environment; Aurora predicts large-scale environmental fields. These roles "
                    "must remain separate, and Aurora verification status must be stated."
                ),
            }, "environmental_evidence", ["era5_reanalysis", "aurora_forecast", "verification"])

        return self._call("foundation.case", {"event_id": resolved}, load_case)

    def live_preflight(self, **kwargs: Any) -> dict[str, Any]:
        """Normalize a time/location request for live ERA5/Aurora analysis."""
        return self._adapt(
            self._call("foundation.live_preflight", kwargs, live_preflight),
            "live_request",
            ["request_normalized"],
        )

    def fetch_era5(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias returning the compact event-time ERA5 snapshot."""
        return self.era5_snapshot(**kwargs)

    def era5_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        """Return compact event-time ERA5 values under surface and pressure_levels."""
        return self._adapt(
            self._call("foundation.era5_snapshot", kwargs, era5_snapshot),
            "era5_event_snapshot",
            ["era5_reanalysis", "event_snapshot"],
        )

    def era5_timeseries(self, **kwargs: Any) -> dict[str, Any]:
        """Return the hourly target-point ERA5 series and exact coverage metadata."""
        return self._adapt(
            self._call("foundation.era5_timeseries", kwargs, era5_timeseries),
            "era5_target_timeseries",
            ["era5_reanalysis", "target_timeseries"],
        )

    def forecast_aurora(self, **kwargs: Any) -> dict[str, Any]:
        """Fetch live ERA5 inputs and run AuroraPretrained when available."""
        return self._adapt(
            self._call("foundation.forecast_aurora", kwargs, forecast_aurora),
            "aurora_forecast",
            ["aurora_forecast", "forecast_verification"],
        )

    def region_fields(self, **kwargs: Any) -> dict[str, Any]:
        """Return aligned ERA5/Aurora regional grids without fixed screening."""
        return self._adapt(
            self._call("foundation.region_fields", kwargs, region_fields),
            "regional_environment_fields",
            ["era5_reanalysis", "aurora_forecast", "regional_grid"],
        )
