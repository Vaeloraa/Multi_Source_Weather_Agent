"""Runtime data and tool contracts exposed to generated Agent programs.

The contract is deliberately descriptive rather than prescriptive: it fixes names,
types and dimensions, while leaving the Agent free to choose fields/statistics.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


SCHEMA_VERSION = "sevir-contract-2"


class SchemaContractError(ValueError):
    """Raised when a semantic field is not provided by an adapter contract."""


def domain_result_error(
    result: Any,
    calls: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, str] | None:
    """Recognize an Agent program that ran but explicitly reported failure.

    Process success and analytical success are separate states.  Without this
    boundary, ``return {"status": "error", ...}`` was counted as a successful
    observation and reset the Agent retry history.
    """
    explicit_status = (
        str(result.get("status", "")).strip().lower()
        if isinstance(result, Mapping)
        else ""
    )
    if explicit_status in {"error", "failed"}:
        message = result.get("message") or result.get("error") or result.get("reason")
        return {
            "error_type": "DomainResultError",
            "error": str(message or f"generated run() returned status={explicit_status}"),
        }
    # Generated code may call a tool that returns a structured error and then
    # merely wrap that object inside another mapping.  Process execution still
    # succeeded, but the analytical action did not.  An explicit top-level
    # status=ok is the only way for generated code to state that it knowingly
    # handled an unavailable call through a successful fallback.
    failed_calls = [
        item
        for item in (calls or [])
        if isinstance(item, Mapping)
        and str(item.get("status", "")).strip().lower()
        in {"error", "failed", "unavailable"}
    ]
    if failed_calls and explicit_status != "ok":
        latest = failed_calls[-1]
        api = str(latest.get("api") or "tool")
        result_status = str(latest.get("result_status") or latest.get("status"))
        error = latest.get("error")
        if isinstance(error, Mapping):
            error = error.get("message") or error.get("type")
        return {
            "error_type": "ToolResultError",
            "error": str(error or f"{api} returned {result_status}"),
        }
    return None


class SemanticRecord:
    """Read evidence by declared semantic names, never guessed raw keys.

    Adapters normalize aliases before this object is created.  Missing values
    therefore produce a precise contract error instead of a chain of speculative
    ``fields``/``values``/``event_id`` accesses in generated code.
    """
    def __init__(self, record: Mapping[str, Any]):
        self._record = record
        self._variables = record.get("variables", {}) if isinstance(record, Mapping) else {}
        if not isinstance(self._variables, Mapping):
            self._variables = {}

    def has(self, name: str) -> bool:
        return str(name) in self._variables

    def variable(self, name: str) -> Any:
        key = str(name)
        if key not in self._variables:
            raise SchemaContractError(f"semantic variable unavailable: {key}")
        value = self._variables[key]
        return value.get("data") if isinstance(value, Mapping) and "data" in value else value

    def metadata(self, name: str, default: Any = None) -> Any:
        value = self._variables.get(str(name), default)
        return value.get(name) if isinstance(value, Mapping) else default

    def to_dict(self) -> dict[str, Any]:
        return dict(self._record) if isinstance(self._record, Mapping) else {}


# Result contracts are registered by shape, rather than embedded in a
# particular model adapter.  A future foundation model can register another
# rule without changing the executor or the Agent repair loop.
RESULT_CONTRACTS: list[tuple[str, Callable[[Mapping[str, Any]], bool], Callable[[dict[str, Any]], None]]] = []


def register_result_contract(
    name: str,
    matches: Callable[[Mapping[str, Any]], bool],
    enrich: Callable[[dict[str, Any]], None],
) -> None:
    RESULT_CONTRACTS.append((str(name), matches, enrich))


class ContractDict(dict):
    """JSON-compatible object with stable attribute access for Agent code.

    It deliberately remains a dict so the HTTP/UI boundary and old clients keep
    working, while generated programs can use Zephyrus-like ``record.field``
    access instead of guessing nested string keys.
    """

    _KEY_ATTRIBUTES = frozenset({"values", "items", "keys"})

    def __getattribute__(self, name: str) -> Any:
        if name in ContractDict._KEY_ATTRIBUTES:
            try:
                return dict.__getitem__(self, name)
            except KeyError:
                pass
        return dict.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def contractize(value: Any) -> Any:
    if isinstance(value, ContractDict):
        return value
    if isinstance(value, Mapping):
        return ContractDict({str(k): contractize(v) for k, v in value.items()})
    if isinstance(value, list):
        return [contractize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(contractize(item) for item in value)
    return value


def classify_execution_error(error_type: str, message: str) -> dict[str, Any]:
    """Return a generic, traceback-driven repair envelope.

    Repair behavior is intentionally not a growing table of message-specific
    patches. The model receives the complete traceback plus the live schema and
    must rewrite the program using that evidence.
    """
    kind = str(error_type or "ExecutionError")
    category = {
        "SyntaxError": "program",
        "NameError": "program",
        "KeyError": "contract",
        "AttributeError": "contract",
        "TypeError": "operation",
        "ValueError": "operation",
        "IndexError": "dimensions",
        "MemoryError": "resource",
        "TimeoutError": "resource",
    }.get(kind, "runtime")
    lowered = str(message).lower()
    if any(token in lowered for token in ("program must contain", "model did not return", "syntax is not allowed")):
        category = "protocol"
    elif any(token in lowered for token in ("'fields'", "'event_id'", "'values'", "'label'", "nonetype", "not subscriptable")):
        category = "contract"
    return {
        "category": category,
        "action": "rewrite the failing program using the complete traceback and the actual runtime_schema/result_schema; change the computation strategy, not just formatting",
        "must_change": True,
        "message": str(message)[:2000],
    }


def normalize_tool_result(tool_api: str, result: Any) -> Any:
    """Expose one stable envelope for every mapping returned by a tool.

    Legacy adapter fields remain at the top level for compatibility, but Agent
    programs have exactly one documented read root: ``result.data``.  Keeping
    this normalization at the tool boundary prevents the model from having to
    guess whether a particular adapter returned ``frames``, ``records`` or a
    model-specific sibling object.
    """
    if not isinstance(result, dict):
        return result
    normalized = contractize(result)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    domain_status = normalized.get("status", "ok")
    normalized.setdefault("result_status", domain_status)
    normalized.setdefault(
        "available",
        domain_status in {"ok", "planned", "normalized"},
    )
    existing_data = normalized.get("data")
    if isinstance(existing_data, Mapping):
        data = contractize(dict(existing_data))
    elif existing_data is not None:
        data = ContractDict({"value": contractize(existing_data)})
    else:
        data = ContractDict()
    normalized["data"] = data

    if "forecast_frames" in normalized:
        normalized.setdefault("data_kind", "forecast_grid")
        normalized.setdefault("dimensions", ["time", "row", "column"])
        data.setdefault("values", normalized["forecast_frames"])
        data.setdefault("relative_minutes", normalized.get("forecast_relative_minutes", []))
        horizon = normalized.get("horizon_minutes")
        if horizon is None and not isinstance(normalized.get("lead_minutes"), (list, tuple)):
            horizon = normalized.get("lead_minutes", 0)
        data.setdefault("horizon_minutes", horizon or 0)
    elif "prediction_frames" in normalized and "truth_frames" in normalized:
        normalized.setdefault("data_kind", "verification_pair")
        normalized.setdefault("dimensions", ["time", "row", "column"])
        data.setdefault("prediction", normalized["prediction_frames"])
        data.setdefault("truth", normalized["truth_frames"])
        data.setdefault("valid_mask", normalized.get("valid_mask"))
        data.setdefault("lead_minutes", normalized.get("lead_minutes", []))
    elif "values" in normalized and "shape" in normalized:
        normalized.setdefault("data_kind", "encoded_grid")
        normalized.setdefault("dimensions", ["row", "column"])
        data.setdefault("values", normalized["values"])

    # Canonical mappings shared by all current adapters.  These assignments
    # reuse the original objects; large grids are not copied.
    if "frames" in normalized:
        normalized.setdefault("data_kind", "observation_grid")
        normalized.setdefault("dimensions", ["time", "row", "column"])
        data.setdefault("values", normalized["frames"])
    elif "values" in normalized:
        data.setdefault("values", normalized["values"])
    for key in ("relative_minutes", "frame_statistics", "missing_value", "valid_mask"):
        if key in normalized:
            data.setdefault(key, normalized[key])
    # Time coordinates are sequences. Legacy scalar values are retained only
    # at their old top-level key and never mirrored into the canonical data
    # coordinate.
    if isinstance(normalized.get("lead_minutes"), (list, tuple)):
        data.setdefault("lead_minutes", normalized["lead_minutes"])
    if "records" in normalized and "ensemble_members" in normalized:
        normalized.setdefault("data_kind", "ensemble_forecast")
        data.setdefault("members", normalized["records"])
        data.setdefault("member_count", normalized["ensemble_members"])
    ensemble = normalized.get("ensemble")
    if isinstance(ensemble, Mapping):
        # Rolling forecasts already expose their native ensemble under the
        # top-level ``ensemble`` object.  The public Agent contract promises a
        # single read root, so mirror that object into ``data`` without copying
        # any of the member grids.  Previously the schema documented
        # ``result.data.members`` but the adapter never created it, forcing the
        # model to guess several incompatible legacy paths.
        members = ensemble.get("members")
        if members is not None:
            normalized["data_kind"] = "ensemble_forecast"
            data.setdefault("members", members)
            member_count = ensemble.get("member_count")
            if member_count is None:
                try:
                    member_count = len(members)
                except TypeError:
                    member_count = 0
            data.setdefault("member_count", member_count)
    if "ground_truth" in normalized:
        data.setdefault("ground_truth", normalized["ground_truth"])
    if "prediction_frames" in normalized:
        data.setdefault("prediction", normalized["prediction_frames"])
    if "truth_frames" in normalized:
        data.setdefault("truth", normalized["truth_frames"])
    if all(key in normalized for key in ("latitude", "longitude")):
        normalized.setdefault("data_kind", "regional_environment")
        for key in ("latitude", "longitude", "era5", "aurora"):
            if key in normalized:
                data.setdefault(key, normalized[key])
    for key in ("artifact_id", "image_url"):
        if key in normalized:
            normalized.setdefault("data_kind", "artifact")
            data.setdefault(key, normalized[key])
    if "event_id" in normalized:
        data.setdefault("event_id", normalized["event_id"])
    if "confidence" in normalized:
        data.setdefault("confidence", normalized["confidence"])
    if not data:
        # Small metadata/search tools vary more than grid adapters.  They still
        # receive the same read root without requiring a bespoke prompt rule.
        for key, value in normalized.items():
            if key not in {"schema_version", "status", "result_status", "available", "data_kind", "data"}:
                data.setdefault(key, value)
    data.setdefault("dimensions", normalized.get("dimensions", []))
    normalized.setdefault("data_kind", "tool_result")
    normalized.setdefault("meta", ContractDict({
        "tool_api": str(tool_api),
        "cache_hit": bool(normalized.get("cache_hit", False)),
    }))
    applied: list[str] = []
    for name, matches, enrich in RESULT_CONTRACTS:
        if matches(normalized):
            enrich(normalized)
            applied.append(name)
    if applied:
        normalized.setdefault("contracts_applied", applied)
    return contractize(normalized)


def _register_builtin_result_contracts() -> None:
    register_result_contract(
        "forecast-grid",
        lambda value: "forecast_frames" in value,
        lambda value: value.setdefault("forecast", {
            "frames": value["forecast_frames"],
            "lead_minutes": list(value.get("forecast_relative_minutes", [])),
            "horizon_minutes": int(
                value.get("horizon_minutes")
                or (
                    value.get("lead_minutes", 0)
                    if not isinstance(value.get("lead_minutes"), (list, tuple))
                    else (value.get("lead_minutes") or [0])[-1]
                )
                or 0
            ),
        }),
    )
    register_result_contract(
        "ensemble-records",
        lambda value: "records" in value and "ensemble_members" in value,
        lambda value: value.setdefault("ensemble", {
            "members": value["records"],
            "member_count": int(value.get("ensemble_members", 0) or 0),
        }),
    )
    register_result_contract(
        "prediction-comparison",
        lambda value: "prediction_frames" in value and "truth_frames" in value,
        lambda value: value.setdefault("comparison", {
            "prediction_frames": value["prediction_frames"],
            "truth_frames": value["truth_frames"],
            "lead_minutes": (
                list(value.get("lead_minutes", []))
                if isinstance(value.get("lead_minutes", []), (list, tuple))
                else [int(value["lead_minutes"])]
            ),
            "valid_mask": value.get("valid_mask"),
        }),
    )


_register_builtin_result_contracts()


def _array_schema(value: Any) -> dict[str, Any]:
    # Generic VIL compatibility keeps the established 3-D meaning
    # ``[frame,row,column]``.  Four-dimensional ensemble payloads add a
    # leading member axis.  Domain-specific arrays must provide explicit
    # ``dimensions`` metadata rather than being guessed here.
    dimension_names = ("member", "frame", "row", "column")
    dimensions = list(dimension_names[-value.ndim:]) if value.ndim else []
    return {
        "type": "ndarray",
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "dimensions": dimensions,
        "elements": int(value.size),
        "read_only": not bool(value.flags.writeable),
    }


def describe_value(value: Any, *, path: str = "", depth: int = 0) -> dict[str, Any]:
    """Return a bounded, machine-readable description without copying arrays."""
    if depth > 5:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, np.ndarray):
        return _array_schema(value)
    if isinstance(value, np.generic):
        return {"type": type(value.item()).__name__, "dtype": str(value.dtype)}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool"}
    if isinstance(value, (int, float, str)):
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        fields: dict[str, Any] = {}
        for key, item in value.items():
            fields[str(key)] = describe_value(item, path=f"{path}.{key}", depth=depth + 1)
        return {"type": "object", "fields": fields}
    if isinstance(value, (list, tuple)):
        result: dict[str, Any] = {"type": "array", "length": len(value)}
        if value:
            result["items"] = describe_value(value[0], path=f"{path}[]", depth=depth + 1)
        return result
    return {"type": type(value).__name__}


def describe_dataset(ds: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(ds):
        record = {
            "index": index,
            "source": item.get("source"),
            "role": item.get("role"),
            "dataset_id": item.get("dataset_id"),
            "status": item.get("status", "available"),
            "capabilities": item.get("capabilities", []),
            "is_primary": bool(item.get("is_primary", False)),
            "kind": item.get("kind"),
            "event_id": item.get("event_id"),
            "upload_id": item.get("upload_id"),
            "shape": item.get("shape"),
            "dtype": item.get("dtype"),
            "source_dtype": item.get("source_dtype"),
            "missing_value_code": item.get("missing_value_code"),
            "fields": {},
        }
        for key in ("values", "raw_data", "relative_minutes", "dimensions", "metadata"):
            if key in item:
                record["fields"][key] = describe_value(item[key], path=f"ds[{index}].{key}")
        records.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "object": "ds",
        "type": "list[DatasetRecord]",
        "records": records,
    }


def tool_method_schema(tool: str, method: str) -> dict[str, Any]:
    """Stable return contract for the raw-grid methods most often used in analysis."""
    if tool == "arrays" and method == "asarray":
        return {"type": "ndarray", "shape": "same as input", "dtype": "float32 by default"}
    if tool == "arrays" and method == "where":
        return {
            "type": "ndarray or tuple[ndarray]",
            "arguments": {
                "where(condition)": "coordinates",
                "where(condition, x, y)": "selected values; x and y are both required",
            },
        }
    if tool == "arrays" and method == "select":
        return {"type": "ndarray[matching_values]", "arguments": {"values": "array", "condition": "same-shaped bool array"}}
    if (
        (tool == "cascast" and method in {"forecast", "forecast_upload"})
        or (tool == "forecaster" and method in {"rolling_raw", "uploaded_sequence", "uploaded_values"})
    ):
        return {
            "type": "dict[str, Any]",
            "read_root": "result.data",
            "fields": {
                "data.values": "ndarray[lead,row,column] (ensemble mean)",
                "data.members": "list[EnsembleMemberRecord] when available",
                "data.members[].frames": "ndarray[lead,row,column]",
                "data.members[].lead_minutes": "list[int] aligned with frames",
                "data.relative_minutes": "list[int]",
                "data.horizon_minutes": "int",
                "data.ground_truth": "dict with available, frames, lead_minutes, source, event_id",
            },
        }
    if (
        (tool == "sevir" and method in {"observations", "historical_observations"})
        or (tool == "data" and method == "load_hdf5")
    ):
        return {
            "type": "dict[str, Any]",
            "read_root": "result.data",
            "fields": {
                "data.values": "ndarray[time,row,column] available to generated code",
                "data.relative_minutes": "list[int] aligned with frames",
                "data.frame_statistics": "list[dict] aligned with frames",
                "data.missing_value": "int (255)",
            },
        }
    if (
        (tool == "aurora" and method == "regional_fields")
        or (tool == "foundation" and method == "region_fields")
    ):
        return {
            "type": "dict[str, Any]",
            "read_root": "result.data",
            "fields": {
                "data.latitude": "ndarray[latitude]",
                "data.longitude": "ndarray[longitude]",
                "data.era5.surface": "dict of ndarray[latitude,longitude]",
                "data.era5.pressure_levels[level]": "dict of ndarray[latitude,longitude]",
                "data.aurora.lead_hours": "list[int]",
                "data.aurora.fields[lead].surface": "dict of ndarray[latitude,longitude]",
                "data.aurora.fields[lead].pressure_levels[level]": "dict of ndarray[latitude,longitude]",
            },
        }
    if tool == "uncertainty" and method == "uploaded_raw":
        return {
            "type": "EnsembleResult",
            "read_root": "result.data",
            "fields": {
                "data.members": "list[EnsembleMemberRecord]",
                "data.members[].lead_minutes": "list[int]",
                "data.members[].frames": "ndarray[lead,row,column]",
                "data.member_count": "int",
                "data.ground_truth": "GroundTruthResult | null",
            },
        }
    if tool == "metrics":
        return {"type": "MetricsResult", "read_root": "result.data", "fields": {
            "data.prediction": "ndarray[lead,row,column]",
            "data.truth": "ndarray[lead,row,column]",
            "data.valid_mask": "ndarray[lead,row,column] | null",
            "data.lead_minutes": "list[int]",
            "question_selected": "JSON-safe evidence",
        }}
    if tool == "visualizer" and method == "plan_spec":
        return {
            "type": "RenderPlanResult",
            "read_root": "result.data",
            "inputs": {
                "spec.chart_type": "bar | grouped_bar | line | ensemble_range_line | heatmap | difference_map | small_multiples | overlay | trajectory | box",
                "spec.series": "list[{label: str, values: list[number]}] for one-dimensional charts",
                "spec.panels": "list[{label: str, values: ndarray[row,column], cmap?: str}] for spatial charts",
                "spec.labels": "optional list[str] aligned with one-dimensional values",
                "spec.title": "optional str",
            },
            "fields": {
                "data.chart_type": "str",
                "data.render_spec": "validated deferred rendering specification",
                "data.render_deferred": "bool",
            },
        }
    if tool == "visualizer" and method == "render_spec":
        return {
            "type": "ArtifactResult",
            "read_root": "result.data",
            "inputs": {
                "spec.chart_type": "bar | grouped_bar | line | ensemble_range_line | heatmap | difference_map | small_multiples | overlay | trajectory | box",
                "spec.series": "list[{label: str, values: list[number]}] for one-dimensional charts",
                "spec.panels": "list[{label: str, values: ndarray[row,column], cmap?: str}] for spatial charts",
                "spec.labels": "optional list[str] aligned with one-dimensional values",
                "spec.title": "optional str",
            },
            "fields": {
                "data.artifact_id": "str",
                "data.image_url": "str",
            },
        }
    if tool == "visualizer":
        return {"type": "ArtifactResult", "read_root": "result.data", "fields": {"data.artifact_id": "str", "data.image_url": "str"}}
    return {"type": "JSON object", "read_root": "result.data", "fields": {"data": "tool-specific canonical payload"}}


def describe_tools(selected_tools: tuple[str, ...], method_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    from ..data_access.tool_contracts import tool_card
    return {
        "schema_version": SCHEMA_VERSION,
        "result_envelope": {
            "status": "result.status or result.result_status",
            "available": "bool",
            "data_kind": "str",
            "read_root": "result.data",
            "compatibility": "top-level legacy adapter fields may exist but are not the Agent contract",
        },
        "tools": {
            tool: {
                "contract": {
                    "inputs": tool_card(tool).inputs,
                    "output_role": tool_card(tool).output_role,
                    "output_schema": tool_card(tool).output_schema,
                    "retry_policy": tool_card(tool).retry_policy,
                    "cache_policy": tool_card(tool).cache_policy,
                },
                "methods": {
                    method: tool_method_schema(tool, method)
                    for method in method_map.get(tool, ())
                }
            }
            for tool in selected_tools
            if tool != "ds"
        },
    }


def augment_error(error: dict[str, Any], ds: list[dict[str, Any]], selected_tools: tuple[str, ...], method_map: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Attach the real available paths to an execution error."""
    enriched = dict(error)
    enriched["schema_version"] = SCHEMA_VERSION
    enriched["available_schema"] = {
        "dataset": describe_dataset(ds),
        "tools": describe_tools(selected_tools, method_map),
    }
    message = str(enriched.get("error", ""))
    if "KeyError" in str(enriched.get("error_type", "")) or "key" in message.lower():
        enriched["repair_hint"] = "Use only paths listed in available_schema; do not invent a sibling key."
    return enriched
