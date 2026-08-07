from __future__ import annotations

from typing import Any

from .backends import evaluation, forecast

from .base import WorldComponent, resolve_event_id


class CasCastForecaster(WorldComponent):
    """Access cached CasCast predictions and deterministic verification.

    The current cache contains 13 history frames and 12 five-minute forecast
    frames for ten events, with ten ensemble members and aligned ground truth.
    `run` does not launch fresh model inference. Arbitrary uploaded sequences
    currently support a Persistence baseline only.
    """

    def run(
        self,
        event_id: Any,
        threshold: float = 133.0,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Return a 60-minute forecast summary and per-lead verification."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "input_frames": 13,
            "forecast_frames": 12,
        }
        return self._call("forecaster.run", arguments, forecast.run_nowcast)

    def hazards(
        self,
        event_id: Any,
        threshold: float = 133.0,
        minimum_region_pixels: int = 16,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Find thresholded forecast regions, approximate locations and duration.

        VIL thresholding is a strong-echo proxy, not rainfall or an official
        warning. Coordinates are approximate SEVIR-grid locations.
        """
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "lead_minutes": 60,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call("forecaster.hazards", arguments, forecast.forecast_hazards)

    def evaluate_dataset(self, primary_threshold: float = 133.0) -> dict[str, Any]:
        """Return the cached ten-event CasCast/Persistence evaluation report."""
        return self._call(
            "forecaster.evaluate_dataset",
            {"primary_threshold": primary_threshold},
            evaluation.evaluate_cascast_dataset,
        )

    def uploaded_sequence(
        self, upload_id: str, threshold: float = 133.0, minimum_region_pixels: int = 16
    ) -> dict[str, Any]:
        """Analyze a 13-frame upload and run Persistence, not live CasCast."""
        arguments = {
            "upload_id": upload_id,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call(
            "forecaster.uploaded_sequence", arguments, forecast.analyze_uploaded_vil_sequence
        )
