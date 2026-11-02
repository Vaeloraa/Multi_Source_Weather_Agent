from __future__ import annotations

"""Agent-facing deterministic analytics capability."""

from typing import Any, Mapping

from .contracts import AnalysisRequest, AnalysisResult
from .executor import IMPLEMENTED_OPERATORS, execute_analysis
from .schema import DEFAULT_PROFILE_REGISTRY


ANALYTICS_CAPABILITY_SCHEMA_VERSION = "sevir-analytics-capability-v1"


class Analytics:
    """Execute validated analytics requests without acquiring new data.

    Analytics is injected into every Worker because it only transforms data
    already authorized for the current Agent request. It cannot load a dataset
    or widen the Capability Gate allow-list.
    """

    def __init__(self, recorder: Any) -> None:
        self._recorder = recorder

    @staticmethod
    def _contract_snapshot() -> dict[str, Any]:
        """Return metadata without creating a scientific tool observation."""

        return {
            "schema_version": ANALYTICS_CAPABILITY_SCHEMA_VERSION,
            "request_schema_version": "analysis-request-v1",
            "result_schema_version": "analysis-result-v1",
            "operators": sorted(IMPLEMENTED_OPERATORS),
            "profiles": DEFAULT_PROFILE_REGISTRY.snapshot(),
            "acquires_data": False,
        }

    def describe(self) -> dict[str, Any]:
        """Return the exact supported operator/profile contract."""

        with self._recorder.work("analytics.describe", {}, backend=False):
            return self._contract_snapshot()

    def execute(
        self,
        request: AnalysisRequest | Mapping[str, Any],
        datasets: Mapping[str, Any],
    ) -> AnalysisResult:
        """Validate and run requested operators on authorized Worker-local values."""

        compiled = (
            request
            if isinstance(request, AnalysisRequest)
            else AnalysisRequest.from_mapping(request)
        )
        arguments = {
            "schema_version": compiled.schema_version,
            "request_id": compiled.request_id,
            "input_handles": [item.data_handle for item in compiled.inputs],
            "operators": [item.op for item in compiled.operators],
        }
        with self._recorder.work("analytics.execute", arguments, backend=False):
            return execute_analysis(compiled, datasets)


__all__ = ["ANALYTICS_CAPABILITY_SCHEMA_VERSION", "Analytics"]
