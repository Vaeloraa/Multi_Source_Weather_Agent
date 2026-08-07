from __future__ import annotations

import inspect
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..backends.evaluation import evaluate_cascast_dataset
from ..backends.forecast import (
    analyze_ensemble_uncertainty,
    analyze_hazard,
    analyze_threshold,
    analyze_uploaded_vil,
    analyze_uploaded_vil_sequence,
    forecast_hazards,
    get_event_summary,
    run_nowcast,
)
from ..backends.geospatial import geolocate_sevir_event
from ..backends.statistics import (
    compare_event_with_history,
    get_noaa_storm_reference,
    historical_sevir_statistics,
)
from ..data_access.catalog import (
    get_sevir_event_metadata,
    load_sevir_vil_event,
    search_sevir_catalog,
)
from .status import run_satellite_to_radar, run_simulator


TOOL_FUNCTIONS = {
    function.__name__: function
    for function in (
        get_sevir_event_metadata,
        load_sevir_vil_event,
        search_sevir_catalog,
        historical_sevir_statistics,
        geolocate_sevir_event,
        get_noaa_storm_reference,
        compare_event_with_history,
        analyze_uploaded_vil,
        analyze_uploaded_vil_sequence,
        get_event_summary,
        analyze_threshold,
        analyze_hazard,
        analyze_ensemble_uncertainty,
        evaluate_cascast_dataset,
        run_nowcast,
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
    return function(**arguments)
