from __future__ import annotations

import json
from typing import Any

from .data import SevirDataIndexer
from .forecast import CasCastForecaster
from .geolocation import SevirGeolocator
from .metrics import VerificationMetrics
from .statistics import SevirStatistics
from .uncertainty import UncertaintyAnalyzer
from .visualization import Visualizer


API_REFERENCE = """SEVIRWorld approved API (all outputs are JSON dictionaries):
world.data.event(event_id)
world.data.search(query='', year=None, month=None, event_type='', limit=10)
world.data.load_hdf5(event_id, start_frame=0, end_frame=49)
world.data.cascast_cache(event_id)
world.data.uploads()
world.geolocator.locate(event_id, row=None, column=None)
world.geolocator.pixel_to_latlon(event_id, row, column)
world.geolocator.reverse_geocode(latitude, longitude)
world.geolocator.locate_admin(event_id, row=192, column=192)
world.forecaster.run(event_id, threshold=133, model_name='cascast')
world.forecaster.hazards(event_id, threshold=133, minimum_region_pixels=16, model_name='cascast')
world.forecaster.evaluate_dataset(primary_threshold=133)
world.forecaster.uploaded_sequence(upload_id, threshold=133, minimum_region_pixels=16)
world.statistics.historical(year=None, event_type='')
world.statistics.compare_event(event_id)
world.statistics.compare_storm_types(event_types=None, year=None)
world.uncertainty.analyze(event_id, threshold=133, minimum_region_pixels=16)
world.uncertainty.track_objects(event_id, source='ensemble_mean', member=1, threshold=133, minimum_region_pixels=16)
world.uncertainty.match_ensemble_objects(event_id, threshold=133, minimum_region_pixels=16)
world.metrics.evaluate_event(event_id, thresholds=None, fss_scales=None)
world.metrics.evaluate_dataset(thresholds=None, fss_scales=None)
world.visualizer.lead_time_metrics(event_id, threshold=133, metric='csi')
world.visualizer.storm_tracks(event_id, source='ensemble_mean', threshold=133)
world.visualizer.storm_type_comparison(event_types=None, year=None)
world.visualizer.ensemble_consensus(event_id, threshold=133)

Rules: never claim an upload used live CasCast; 255 is missing ground truth;
VIL thresholds are encoded-value proxies, not rainfall or official warnings;
member support is not calibrated probability; catalog statistics are metadata.
Assign the final JSON-serializable analysis to a variable named result.
"""


def _trace_result(result: Any) -> Any:
    try:
        encoded = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"type": type(result).__name__}
    if len(encoded) <= 40_000:
        return result
    if isinstance(result, dict):
        return {
            key: result.get(key)
            for key in (
                "status",
                "task",
                "event_id",
                "scope",
                "chart_type",
                "artifact_id",
                "image_url",
                "interpretation_limit",
            )
            if key in result
        } | {"trace_truncated": True, "original_json_characters": len(encoded)}
    return {"trace_truncated": True, "original_json_characters": len(encoded)}


class SevirWorld:
    """Facade injected into the restricted code-agent process.

    `context` contains server-bound event/upload/threshold values. Components
    expose only high-level APIs; arbitrary filesystem, network and process
    access are intentionally absent. Every call is recorded for provenance.
    """

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self.context = dict(context or {})
        self._calls: list[dict[str, Any]] = []
        self.data = SevirDataIndexer(self._record)
        self.geolocator = SevirGeolocator(self._record)
        self.forecaster = CasCastForecaster(self._record)
        self.statistics = SevirStatistics(self._record)
        self.uncertainty = UncertaintyAnalyzer(self._record)
        self.metrics = VerificationMetrics(self._record)
        self.visualizer = Visualizer(
            self._record, self.metrics, self.statistics, self.uncertainty
        )

    @staticmethod
    def api_reference() -> str:
        """Return the exact API contract shown to the code-generating model."""
        return API_REFERENCE

    def _record(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        self._calls.append(
            {
                "step": len(self._calls) + 1,
                "api": name,
                "arguments": arguments,
                "result": result,
                "trace_result": _trace_result(result),
            }
        )

    def call_trace(self) -> list[dict[str, Any]]:
        """Return bounded, JSON-safe call records for the execution trace."""
        return [
            {
                "step": item["step"],
                "api": item["api"],
                "arguments": item["arguments"],
                "result": item["trace_result"],
            }
            for item in self._calls
        ]

    def primary_result(self) -> dict[str, Any]:
        """Choose the richest result for compatibility with the existing UI."""
        if not self._calls:
            return {}
        priorities = {
            "cascast_dataset_evaluation": 100,
            "multi_threshold_verification": 95,
            "ensemble_uncertainty_analysis": 90,
            "cross_ensemble_object_matching": 85,
            "future_hazard_analysis": 80,
            "nowcast": 70,
            "storm_object_tracking": 65,
            "uploaded_sevir_vil_sequence_analysis": 60,
            "uploaded_sevir_vil_analysis": 60,
            "sevir_chart": 50,
        }
        candidates = [item["result"] for item in self._calls if isinstance(item["result"], dict)]
        return max(candidates, key=lambda item: priorities.get(str(item.get("task", "")), 10))
