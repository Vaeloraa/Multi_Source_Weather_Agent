"""Model-neutral contracts for injected tools.

Adapters may expose different internal objects, but the Agent sees one registry
describing inputs, semantic output role, cache policy and retry class.  This is
metadata, not a second execution path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCard:
    name: str
    inputs: dict[str, str] = field(default_factory=dict)
    output_role: str = "analysis"
    output_schema: dict[str, str] = field(default_factory=dict)
    error_classes: tuple[str, ...] = ()
    retry_policy: str = "transient_only"
    cache_policy: str = "request_key"


TOOL_CARDS: dict[str, ToolCard] = {
    "sevir": ToolCard("sevir", {"event_or_location": "bound identity or explicit query"}, "observation_access", {"data": "SEVIR metadata or raw observations"}, ("not_found", "schema"), "event_or_query"),
    "sevir.observations": ToolCard("sevir.observations", {"event_id": "optional bound string", "frame_range": "int,int"}, "local_observation", {"data.values": "array[time,row,column]", "data.relative_minutes": "list[int]"}, ("not_found", "schema"), "event_hdf5"),
    "sevir.similar_events": ToolCard("sevir.similar_events", {"event_id": "optional bound string", "field": "string"}, "historical_candidates", {"data": "candidate metadata"}, ("not_found", "schema"), "catalog"),
    "geolocator": ToolCard("geolocator", {"event_or_coordinates": "bound event or WGS84"}, "geospatial_mapping", {"data": "coordinates and administrative polygons"}, ("not_found", "schema"), "deterministic"),
    "cascast": ToolCard("cascast", {"event_or_upload": "bound identity", "horizon_hours": "1|2|3"}, "local_forecast", {"data.values": "array[lead,row,column]", "data.members": "ensemble grids"}, ("resource", "model", "schema"), "model_identity"),
    "aurora": ToolCard("aurora", {"event_or_time_region": "bound event or explicit UTC/region"}, "regional_environment", {"data": "ERA5/Aurora evidence or grids"}, ("network", "resource", "model", "schema"), "request_identity"),
    "foundation.live_preflight": ToolCard("foundation.live_preflight", {"time": "datetime", "location": "point_or_region"}, "request", {"data.request_key": "string", "result_status": "string"}, ("invalid_request", "credential", "network")),
    "foundation.fetch_era5": ToolCard("foundation.fetch_era5", {"request_key": "string"}, "environment_reanalysis", {"data": "standardized_environment"}, ("network", "credential", "schema")),
    "foundation.forecast_aurora": ToolCard("foundation.forecast_aurora", {"era5": "standardized_environment", "leads": "list[int]"}, "environment_forecast", {"data": "forecast_fields"}, ("resource", "schema", "model")),
    "foundation.region_fields": ToolCard("foundation.region_fields", {"request_key": "string"}, "regional_environment", {"data.latitude": "array", "data.longitude": "array", "data.era5": "regional_fields", "data.aurora": "regional_fields"}, ("resource", "schema", "model")),
    "data.load_hdf5": ToolCard("data.load_hdf5", {"event_id": "string", "start_frame": "int", "end_frame": "int"}, "local_observation", {"data.values": "array[time,row,column]", "data.relative_minutes": "list[int]", "data.frame_statistics": "list[object]", "data.missing_value": "int"}, ("not_found", "schema"), "event_npz_or_hdf5"),
    "data.match_uploaded_event": ToolCard("data.match_uploaded_event", {"upload_id": "string"}, "event_binding", {"data.event_id": "string", "data.confidence": "number"}, ("not_found",), "none"),
    "forecaster": ToolCard("forecaster", {"event_id": "string", "lead_minutes": "list[int]"}, "local_forecast", {"data.values": "grid", "data.ground_truth": "optional"}, ("resource", "model")),
    "metrics": ToolCard("metrics", {"prediction": "array", "truth": "array"}, "verification", {"data": "verification evidence"}, ("schema", "dimensions"), "none"),
    "visualizer": ToolCard("visualizer", {"data": "evidence"}, "presentation", {"data.artifact_id": "string"}, ("schema", "render"), "none"),
}


def tool_card(name: str) -> ToolCard:
    key = str(name)
    if key in TOOL_CARDS:
        return TOOL_CARDS[key]
    canonical = key.split(".", 1)[0]
    return TOOL_CARDS.get(canonical, ToolCard(key))


def tool_card_snapshot() -> dict[str, dict[str, Any]]:
    return {name: asdict(card) for name, card in TOOL_CARDS.items()}


def classify_tool_error(error: Exception | str) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    if any(token in lowered for token in ("program must contain", "model did not return", "syntax is not allowed")):
        category, retry = "protocol", "do_not_repeat_same_program"
    elif any(token in lowered for token in ("fields", "event_id", "values", "label", "noneType", "not subscriptable")):
        category, retry = "schema", "adapter_or_contract_repair_once"
    elif any(token in lowered for token in ("timeout", "timed out", "connection", "temporary", "503", "429")):
        category, retry = "transient", "retry_with_backoff"
    else:
        category, retry = "runtime", "inspect_then_retry_once"
    return {"category": category, "retry_policy": retry, "message": message[:2000]}
