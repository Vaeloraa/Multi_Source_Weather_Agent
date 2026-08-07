from __future__ import annotations

import json
from typing import Any, Callable

from .contracts import CANONICAL_ARGUMENTS, ExecutionRequest
from .data import SevirDataIndexer
from .forecast import CasCastForecaster
from .geolocation import SevirGeolocator
from .metrics import VerificationMetrics
from .statistics import SevirStatistics
from .uncertainty import UncertaintyAnalyzer
from .visualization import Visualizer


TOOL_CLASSES = {
    "geolocator": SevirGeolocator,
    "forecaster": CasCastForecaster,
    "statistics": SevirStatistics,
    "uncertainty": UncertaintyAnalyzer,
    "metrics": VerificationMetrics,
    "visualizer": Visualizer,
}


def _trace_result(result: Any) -> Any:
    try:
        encoded = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"type": type(result).__name__}
    if len(encoded) <= 40_000:
        return result
    if isinstance(result, dict):
        keys = (
            "status", "task", "event_id", "scope", "chart_type", "artifact_id",
            "image_url", "interpretation_limit",
        )
        return {key: result.get(key) for key in keys if key in result} | {
            "trace_truncated": True,
            "original_json_characters": len(encoded),
        }
    return {"trace_truncated": True, "original_json_characters": len(encoded)}


class ToolUsageRecorder:
    """Record bounded results and failures for every injected tool call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, name: str, arguments: dict[str, Any], result: Any = None, error: Exception | None = None
    ) -> None:
        item: dict[str, Any] = {
            "step": len(self.calls) + 1,
            "api": name,
            "arguments": arguments,
            "status": "error" if error else "ok",
        }
        if error:
            item["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            item["result"] = _trace_result(result)
        self.calls.append(item)


class DataIndexer:
    """Server-side loader that resolves data_descs before generated code runs."""

    def load(self, data_descs: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        loader = SevirDataIndexer(lambda *_: None)
        datasets: list[dict[str, Any]] = []
        for description in data_descs:
            source = str(description.get("source", "sevir")).lower()
            if source == "sevir":
                event_id = str(description.get("event_id", "")).strip()
                if not event_id:
                    raise ValueError("SEVIR data_desc requires event_id")
                metadata = loader.event(event_id)
                datasets.append(
                    {"source": "sevir", "event_id": event_id, "metadata": metadata["record"]}
                )
            elif source == "upload":
                upload_id = str(description.get("upload_id", "")).strip()
                records = loader.uploads().get("records", [])
                metadata = next((item for item in records if item.get("upload_id") == upload_id), None)
                if metadata is None:
                    raise ValueError(f"unknown upload_id: {upload_id}")
                datasets.append({"source": "upload", "upload_id": upload_id, "metadata": metadata})
            else:
                raise ValueError(f"unsupported data source: {source}")
        return datasets


class ExecutionEnvironment:
    """Assemble request-scoped resources in the canonical Zephyrus order."""

    def __init__(self, request: ExecutionRequest) -> None:
        self.request = request
        self.recorder = ToolUsageRecorder()
        metrics = VerificationMetrics(self.recorder)
        statistics = SevirStatistics(self.recorder)
        uncertainty = UncertaintyAnalyzer(self.recorder)
        self.resources: dict[str, Any] = {
            "ds": DataIndexer().load(request.data_descs),
            "geolocator": SevirGeolocator(self.recorder),
            "forecaster": CasCastForecaster(self.recorder),
            "statistics": statistics,
            "uncertainty": uncertainty,
            "metrics": metrics,
            "visualizer": Visualizer(self.recorder, metrics, statistics, uncertainty),
        }

    def arguments(self) -> list[Any]:
        return [self.resources[name] for name in CANONICAL_ARGUMENTS if name in self.request.selected_tools]

    def invoke(self, run_function: Callable[..., Any]) -> Any:
        return run_function(*self.arguments())
