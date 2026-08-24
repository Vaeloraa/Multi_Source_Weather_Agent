from __future__ import annotations

import inspect
import time
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..backends.evaluation import evaluate_cascast_dataset
from ..backends.forecast import (
    analyze_ensemble_uncertainty,
    analyze_hazard,
    analyze_threshold,
    analyze_uploaded_vil,
    get_uploaded_vil_values,
    get_uploaded_vil_sequence_values,
    get_uploaded_vil_sequence_forecast,
    analyze_uploaded_vil_sequence,
    forecast_hazards,
    get_event_summary,
    run_nowcast,
    analyze_area_ratio,
    analyze_duration,
    analyze_object_tracking,
    analyze_historical_anomaly,
    analyze_composite_hazard,
    analyze_physical_area,
)
from ..backends.geospatial import geolocate_pixel_region, geolocate_sevir_event, reverse_geocode_us
from ..backends.statistics import (
    aggregate_sevir_catalog,
    catalog_schema,
    compare_sevir_catalog_groups,
    compare_event_with_history,
    compare_observed_event_with_historical_window,
    get_noaa_storm_reference,
    group_sevir_catalog,
    historical_sevir_statistics,
    load_event_history_frame_primitives,
    find_similar_event_candidates,
    query_sevir_records,
    query_sevir_values,
    rank_sevir_catalog,
)
from ..data_access.catalog import (
    get_sevir_event_metadata,
    load_sevir_vil_event,
    search_sevir_catalog,
    search_sevir_catalog_by_region,
)
from ..data_access.foundation import get_case_foundation_artifacts, get_model_capabilities
from ..tools.live_foundation import fetch_era5, forecast_aurora, live_preflight
from ..data_access.uploads import match_uploaded_event
from .status import run_satellite_to_radar, run_simulator
from .observability import current_trace


def rolling_forecast(
    event_id: str,
    model_name: str = "cascast_live",
    horizon_hours: float = 2.0,
    threshold: float = 133.0,
    ensemble_members: int = 10,
) -> dict[str, Any]:
    """Run a 2- or 3-hour rolling forecast as consecutive 60-minute windows."""
    horizon = float(horizon_hours)
    if horizon not in {2.0, 3.0}:
        raise ValueError("horizon_hours must be 2.0 or 3.0")
    try:
        return run_nowcast(
            event_id=event_id,
            model_name=model_name,
            threshold=threshold,
            input_frames=13,
            forecast_frames=int(horizon * 12),
            ensemble_members=ensemble_members,
        )
    except RuntimeError as error:
        if "HDF5" in str(error) or "not downloaded" in str(error):
            return {
                "status": "awaiting_assets",
                "task": "rolling_forecast",
                "event_id": event_id,
                "horizon_hours": horizon,
                "required_assets": ["same-event official SEVIR HDF5"],
                "error": str(error),
                "interpretation_limit": "Rolling forecasts require consecutive windows from the same event; unrelated cached events are never concatenated.",
            }
        raise


TOOL_FUNCTIONS = {
    function.__name__: function
    for function in (
        get_sevir_event_metadata,
        load_sevir_vil_event,
        search_sevir_catalog,
        search_sevir_catalog_by_region,
        get_model_capabilities,
        get_case_foundation_artifacts,
        live_preflight,
        fetch_era5,
        forecast_aurora,
        catalog_schema,
        query_sevir_records,
        query_sevir_values,
        aggregate_sevir_catalog,
        group_sevir_catalog,
        compare_sevir_catalog_groups,
        rank_sevir_catalog,
        historical_sevir_statistics,
        load_event_history_frame_primitives,
        find_similar_event_candidates,
        geolocate_sevir_event,
        geolocate_pixel_region,
        match_uploaded_event,
        get_noaa_storm_reference,
        compare_event_with_history,
        compare_observed_event_with_historical_window,
        analyze_uploaded_vil,
        get_uploaded_vil_values,
        get_uploaded_vil_sequence_values,
        get_uploaded_vil_sequence_forecast,
        analyze_uploaded_vil_sequence,
        get_event_summary,
        analyze_threshold,
        analyze_hazard,
        analyze_ensemble_uncertainty,
        evaluate_cascast_dataset,
        run_nowcast,
        rolling_forecast,
        analyze_area_ratio,
        analyze_duration,
        analyze_object_tracking,
        analyze_historical_anomaly,
        analyze_composite_hazard,
        analyze_physical_area,
        forecast_hazards,
        run_satellite_to_radar,
        run_simulator,
    )
}


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        non_null = [item for item in get_args(annotation) if item is not type(None)]
        return _json_type(non_null[0]) if non_null else "string"
    return {int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}.get(
        origin or annotation, "string"
    )


def _tool_schema(function: Any) -> dict[str, Any]:
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": _json_type(hints.get(name, parameter.annotation))}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        elif parameter.default is not None:
            properties[name]["default"] = parameter.default
    description = inspect.getdoc(function) or f"Execute {function.__name__}."
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [_tool_schema(function) for function in TOOL_FUNCTIONS.values()]


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ValueError(f"tool is not registered: {name}")
    trace = current_trace()
    started = time.perf_counter()
    if trace is not None:
        trace.increment("tool_calls", 1)
    try:
        result = function(**arguments)
    except Exception as error:
        if trace is not None:
            trace.record_stage(
                f"tool:{name}",
                (time.perf_counter() - started) * 1000,
                status="error",
                tool_name=name,
                argument_keys=sorted(arguments),
                error_type=type(error).__name__,
                error=str(error)[:500],
            )
        raise
    if trace is not None:
        cache = result.get("cache", {}) if isinstance(result, dict) else {}
        trace.record_stage(
            f"tool:{name}",
            (time.perf_counter() - started) * 1000,
            status="ok",
            tool_name=name,
            argument_keys=sorted(arguments),
            cache_hit=bool(cache.get("hit")) if isinstance(cache, dict) else False,
            result_status=result.get("status") if isinstance(result, dict) else None,
        )
    # An uploaded event match is also the point at which deterministic
    # geospatial evidence becomes available. Attach it immediately so the
    # Agent cannot finish with only an event ID and omit the location.
    if name == "match_uploaded_event" and isinstance(result, dict):
        event_match = result.get("event_match", {})
        event_id = str(event_match.get("event_id", "")).strip()
        if event_match.get("status") == "matched" and event_id:
            location = geolocate_sevir_event(event_id)
            result["event_location"] = location
            result["location_verified"] = True
            center = location.get("approx_center_latlon")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                result["administrative_area"] = reverse_geocode_us(
                    float(center[0]), float(center[1])
                )
        else:
            result["location_verified"] = False
    # Forecast/statistics tools also carry an event_id. Attach the same
    # geospatial evidence to their result so the Agent cannot report a
    # location-free event analysis when the Catalog has usable bounds.
    if isinstance(result, dict) and result.get("event_id") and "event_location" not in result:
        try:
            location = geolocate_sevir_event(str(result["event_id"]))
            result["event_location"] = location
            result["location_verified"] = True
            center = location.get("approx_center_latlon")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                result["administrative_area"] = reverse_geocode_us(
                    float(center[0]), float(center[1])
                )
        except (OSError, RuntimeError, ValueError, KeyError, IndexError):
            result.setdefault("administrative_area", {
                "status": "unresolved",
                "reason": "Catalog bounds were unavailable or administrative lookup failed.",
            })
    return result
