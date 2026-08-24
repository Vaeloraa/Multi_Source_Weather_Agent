from __future__ import annotations

from typing import Any

from ..backends import evaluation, forecast
from ..domain.thresholds import DEFAULT_CASCAST_VIL_THRESHOLD

from .base import WorldComponent, resolve_event_id


class CasCastForecaster(WorldComponent):
    """Run local CasCast inference or access cached predictions and verification.

    The current cache contains 13 history frames and 12 five-minute forecast
    frames for ten events, with ten ensemble members and aligned ground truth.
    Set ``model_name='cascast_live'`` to read an official local HDF5 event and
    run EarthFormer, CasCast diffusion and VAE decoding on the local GPU. Live
    results are cached; ``model_name='cascast'`` uses the ten-event package.
    """

    def run(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        model_name: str = "cascast",
        ensemble_members: int = 10,
    ) -> dict[str, Any]:
        """Return a 60-minute or 2–3 hour rolling forecast summary."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "input_frames": 13,
            "forecast_frames": 12,
            "ensemble_members": ensemble_members,
        }
        return self._call("forecaster.run", arguments, forecast.run_nowcast)

    def rolling(
        self,
        event_id: Any,
        horizon_hours: float = 2.0,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        model_name: str = "cascast_live",
        ensemble_members: int = 10,
    ) -> dict[str, Any]:
        """Run a two- or three-hour rolling forecast in 5-minute frames."""
        horizon = float(horizon_hours)
        if horizon not in {2.0, 3.0}:
            raise ValueError("horizon_hours must be 2.0 or 3.0")
        event_id = resolve_event_id(event_id)
        forecast_frames = int(horizon * 12)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "input_frames": 13,
            "forecast_frames": forecast_frames,
            "ensemble_members": ensemble_members,
        }
        return self._call("forecaster.rolling", arguments, forecast.run_nowcast)

    def rolling_raw(
        self,
        event_id: Any,
        horizon_hours: float = 2.0,
        model_name: str = "cascast_live",
        ensemble_members: int = 10,
        include_ground_truth: bool = False,
    ) -> dict[str, Any]:
        """Return raw ensemble grids with a stable contract.

        ``result['forecast_frames']`` is the mean float32
        ``[lead,row,column]`` grid; ``result['lead_minutes']`` is its aligned
        list. Members are ``result['ensemble']['members']`` records whose
        ``frames`` have the same shape and whose ``lead_minutes`` are aligned.
        ``horizon_minutes`` is scalar. Future truth is excluded unless
        ``include_ground_truth=True`` is explicitly requested for verification;
        it is then nested under ``ground_truth``. No metric, threshold, trend,
        region or interpretation is precomputed.
        """
        horizon = float(horizon_hours)
        if horizon not in {1.0, 2.0, 3.0}:
            raise ValueError("horizon_hours must be 1.0, 2.0 or 3.0")
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "horizon_hours": horizon,
            "ensemble_members": ensemble_members,
            "include_ground_truth": bool(include_ground_truth),
            "array_transport": True,
        }
        return self._call(
            "forecaster.rolling_raw",
            arguments,
            forecast.get_rolling_forecast_raw_values,
        )

    def hazards(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        model_name: str = "cascast",
        ensemble_members: int = 10,
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
            "ensemble_members": ensemble_members,
        }
        return self._call("forecaster.hazards", arguments, forecast.forecast_hazards)

    def area_ratio(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        area_ratio_rule: float = 0.001,
        lead_minutes: int = 60,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Compute per-lead above-threshold area fractions using an explicit rule."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "area_ratio_rule": area_ratio_rule,
            "lead_minutes": lead_minutes,
        }
        return self._call("forecaster.area_ratio", arguments, forecast.analyze_area_ratio)

    def duration(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        area_ratio_rule: float = 0.001,
        lead_minutes: int = 60,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Compute cumulative and longest consecutive duration above an area rule."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "area_ratio_rule": area_ratio_rule,
            "lead_minutes": lead_minutes,
        }
        return self._call("forecaster.duration", arguments, forecast.analyze_duration)

    def object_tracking(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        minimum_region_pixels: int = 1,
        lead_minutes: int = 60,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Track thresholded forecast objects and report splits/associations."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
            "lead_minutes": lead_minutes,
        }
        return self._call(
            "forecaster.object_tracking", arguments, forecast.analyze_object_tracking
        )

    def historical_anomaly(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        quantile: float = 0.9,
        minimum_region_pixels: int = 1,
    ) -> dict[str, Any]:
        """Compare observed history with the available cached-event baseline."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "threshold": threshold,
            "quantile": quantile,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call(
            "forecaster.historical_anomaly", arguments, forecast.analyze_historical_anomaly
        )

    def composite_hazard(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        area_ratio_rule: float = 0.001,
        minimum_consecutive_minutes: int = 15,
        lead_minutes: int = 60,
    ) -> dict[str, Any]:
        """Synthesize area, duration, ensemble and historical evidence."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "threshold": threshold,
            "area_ratio_rule": area_ratio_rule,
            "minimum_consecutive_minutes": minimum_consecutive_minutes,
            "lead_minutes": lead_minutes,
        }
        return self._call(
            "forecaster.composite_hazard", arguments, forecast.analyze_composite_hazard
        )

    def physical_area(
        self,
        event_id: Any,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        lead_minutes: int = 60,
        model_name: str = "cascast",
    ) -> dict[str, Any]:
        """Estimate above-threshold physical area in square kilometres."""
        event_id = resolve_event_id(event_id)
        arguments = {
            "event_id": event_id,
            "model_name": model_name,
            "threshold": threshold,
            "lead_minutes": lead_minutes,
        }
        return self._call(
            "forecaster.physical_area", arguments, forecast.analyze_physical_area
        )

    def evaluate_dataset(self, primary_threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD) -> dict[str, Any]:
        """Return the cached ten-event CasCast/Persistence evaluation report."""
        return self._call(
            "forecaster.evaluate_dataset",
            {"primary_threshold": primary_threshold},
            evaluation.evaluate_cascast_dataset,
        )

    def uploaded_image(
        self,
        upload_id: str,
        threshold: float = DEFAULT_CASCAST_VIL_THRESHOLD,
        minimum_region_pixels: int = 1,
    ) -> dict[str, Any]:
        """Compatibility summary for one uploaded 384x384 VIL image.

        This endpoint is retained for legacy clients. New Agent-generated
        analyses must use ``ds[0]['raw_data']`` (or ``ds[0]['values']``) so fields, thresholds,
        regions, and statistics are selected for the question rather than fixed
        by this compatibility method.
        """
        arguments = {
            "upload_id": upload_id,
            "threshold": threshold,
            "minimum_region_pixels": minimum_region_pixels,
        }
        return self._call(
            "forecaster.uploaded_image", arguments, forecast.analyze_uploaded_vil
        )

    def uploaded_values(self, upload_id: str) -> dict[str, Any]:
        """Return raw pixels from one uploaded VIL image without fixed statistics.

        Generated ``run()`` code must choose and calculate metrics such as
        median, quantiles, trimmed mean, or MAD.
        """
        arguments = {"upload_id": upload_id}
        return self._call(
            "forecaster.uploaded_values", arguments, forecast.get_uploaded_vil_values
        )

    def uploaded_sequence(
        self,
        upload_id: str,
        model_name: str = "cascast_live",
        ensemble_members: int = 10,
        horizon_hours: float = 1.0,
        include_ground_truth: bool = False,
        force_live: bool = False,
    ) -> dict[str, Any]:
        """Run a forecast model and return raw future pixel grids.

        ``horizon_hours`` may be 1.0, 2.0, or 3.0. For CasCast, horizons above
        one hour use recursive rolling conditioning over consecutive 60-minute
        blocks from this same upload. Do not pass threshold or region arguments:
        this tool computes no fixed
        fields or hazard summaries. ``history_frames`` and ``forecast_frames``
        are read-only arrays shaped ``[time,row,column]`` inside the isolated
        Worker. The generated run() program must choose metrics and compare the
        raw float32 forecast frames with raw history. Preserve the returned
        forecast_evidence_id in compact output so follow-ups reuse this run.
        Ground truth is deliberately excluded by default so an ordinary
        forecast cannot use hindsight. Set ``include_ground_truth=True`` only
        when the user explicitly asks to verify accuracy against the observed
        future. If this lower-level path is used, truth is nested at
        ``result['ground_truth']['frames']`` with its times at
        ``result['ground_truth']['lead_minutes']``; forecast times are in
        ``result['forecast_relative_minutes']``. There are no top-level
        ``ground_truth_frames`` or ``truth_frames`` keys. For open-ended error
        calculations prefer ``metrics.uploaded_raw_pair``, which returns an
        already aligned ``prediction_frames``, ``truth_frames`` and
        ``valid_mask``. When an uploaded history exactly matches a local HDF5
        row, its validation manifest determines truth coverage from the exact
        segment: segment 0/1/2 supports at most +180/+120/+60 minutes. Value
        255 is missing.
        Do not call this method for a ``kind='single_frame'`` upload.
        """
        if str(model_name).strip().lower() in {"cascast", "cascast_live"}:
            model_name = "cascast_live"
        arguments = {
            "upload_id": upload_id,
            "model_name": model_name,
            "ensemble_members": ensemble_members,
            "horizon_hours": horizon_hours,
            "include_ground_truth": bool(include_ground_truth),
            "force_live": bool(force_live),
            "array_transport": True,
        }
        return self._call(
            "forecaster.uploaded_sequence",
            arguments,
            forecast.get_uploaded_vil_sequence_forecast,
        )


    def uploaded_history(
        self,
        upload_id: str,
    ) -> dict[str, Any]:
        """Deprecated compatibility endpoint returning raw pixels and timestamps.

        This method deliberately has no ``fields``, ``threshold``, region, or
        tracking arguments.  It performs no statistical calculation.  The
        The preferred path is to inspect ``ds[0]["raw_data"]`` directly. This
        endpoint remains only for older clients and performs no statistics.
        """
        arguments = {"upload_id": upload_id}
        return self._call(
            "forecaster.uploaded_history",
            arguments,
            forecast.get_uploaded_vil_sequence_values,
        )
