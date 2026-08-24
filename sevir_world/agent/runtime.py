from __future__ import annotations

import json
import hashlib
import importlib
import inspect
from typing import Any, Callable

import numpy as np

from .contracts import CANONICAL_ARGUMENTS, ExecutionRequest
from .schema import (
    SCHEMA_VERSION,
    describe_dataset,
    describe_value,
    normalize_tool_result,
)
from ..data_access.tool_contracts import tool_card, classify_tool_error
from .ledger import EvidenceLedger
from ..data_access.uploads import _load_uploaded_vil, _load_uploaded_vil_sequence
from ..data_access.config import ROOT


_TOOL_IMPORTS = {
    "sevir": ("sevir_world.tools.capabilities", "SevirCapability"),
    "geolocator": ("sevir_world.tools.capabilities", "GeolocatorCapability"),
    "cascast": ("sevir_world.tools.capabilities", "CasCastCapability"),
    "aurora": ("sevir_world.tools.capabilities", "AuroraCapability"),
}

# Product-facing names and legacy program names resolve to one component; the
# aliases never create a second backend instance or a duplicate tool record.
TOOL_ALIASES: dict[str, str] = {}


_LOADED_TOOL_CLASSES: dict[str, type[Any]] = {}


def _load_tool_class(name: str) -> type[Any]:
    """Import one concrete tool only when the bound request needs it."""

    canonical = TOOL_ALIASES.get(str(name), str(name))
    if canonical not in _TOOL_IMPORTS:
        raise RuntimeError(f"unknown worker tool: {name}")
    loaded = _LOADED_TOOL_CLASSES.get(canonical)
    if loaded is None:
        module_name, class_name = _TOOL_IMPORTS[canonical]
        loaded = getattr(importlib.import_module(module_name), class_name)
        _LOADED_TOOL_CLASSES[canonical] = loaded
    return loaded


def canonical_tool_name(name: str) -> str:
    return TOOL_ALIASES.get(str(name), str(name))


def _trace_result(result: Any) -> Any:
    def array_metadata(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "array": True,
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "elements": int(value.size),
                "trace_values_omitted": True,
            }
        if isinstance(value, dict):
            return {str(key): array_metadata(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [array_metadata(item) for item in value]
        return value

    summarised = array_metadata(result)
    try:
        encoded = json.dumps(summarised, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"type": type(result).__name__}
    if len(encoded) <= 40_000:
        return summarised
    if isinstance(result, dict):
        keys = (
            "status", "task", "event_id", "scope", "chart_type", "artifact_id",
            "image_url", "url", "mime_type", "title", "description", "size_bytes",
            "verified", "capabilities", "ground_truth_available", "interpretation_limit",
            "upload_id", "kind", "model_name", "output_type", "ensemble_count",
            "ensemble_members", "member_count", "forecast_frame_count", "history_frame_count",
            "lead_minutes", "forecast_relative_minutes", "ground_truth_complete_for_horizon",
            "requested_forecast_leads", "complete_forecast_sequence", "artifact_count", "frame_count",
            "event_match",
            "forecast_evidence_id", "forecast_evidence", "member_seeds", "cache_hit",
            "ground_truth_requested", "ground_truth_attached", "verified_lead_minutes",
        )
        return {key: result.get(key) for key in keys if key in result} | {
            "trace_truncated": True,
            "original_json_characters": len(encoded),
        }
    return {"trace_truncated": True, "original_json_characters": len(encoded)}


def _trace_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducible scalar arguments without echoing full pixel matrices."""
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {"argument_keys": sorted(arguments), "trace_truncated": True}
    if len(encoded) <= 20_000:
        return arguments
    compact: dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            compact[key] = value
        elif key in {"labels", "lead_minutes", "series_labels"}:
            compact[key] = value
        elif isinstance(value, (list, tuple)):
            compact[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            compact[f"{key}_keys"] = sorted(str(item) for item in value)[:30]
    compact["trace_truncated"] = True
    compact["original_json_characters"] = len(encoded)
    return compact


def tool_result_cache_key(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"api": str(name), "arguments": arguments},
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cacheable_tool_result(value: Any, limit: int = 20_000) -> Any | None:
    """Return a lossless JSON copy suitable for parent-host replay."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > limit:
        return None
    return json.loads(encoded)


class ToolUsageRecorder:
    """Record bounded results and failures for every injected tool call."""

    def __init__(self, replay_results: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.ledger = EvidenceLedger()
        self._replay_results = dict(replay_results or {})
        self._reusable_results: dict[str, Any] = {}

    def lookup(self, name: str, arguments: dict[str, Any]) -> Any | None:
        key = tool_result_cache_key(name, arguments)
        if key not in self._replay_results:
            return None
        # The payload crossed a JSON boundary and is already detached from the
        # parent's state. Normalize it exactly like a fresh backend response.
        return normalize_tool_result(name, self._replay_results[key])

    def reusable_results(self) -> dict[str, Any]:
        return dict(self._reusable_results)

    def __call__(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
        cache_hit: bool = False,
    ) -> None:
        card = tool_card(name)
        domain_status = result.get("status") if isinstance(result, dict) else None
        unavailable_statuses = {
            "not_found", "unavailable", "partial", "dependency_missing", "inputs_missing",
            "awaiting_assets", "inference_failed", "extraction_failed", "error", "unsupported",
        }
        execution_status = "unavailable" if domain_status in unavailable_statuses else ("error" if error else "ok")
        item: dict[str, Any] = {
            "step": len(self.calls) + 1,
            "api": name,
            "arguments": _trace_arguments(arguments),
            "status": execution_status,
            "result_status": domain_status,
            "cache_hit": bool(cache_hit),
            "cache_key": tool_result_cache_key(name, arguments),
            "tool_contract": {
                "name": card.name, "output_role": card.output_role,
                "retry_policy": card.retry_policy, "cache_policy": card.cache_policy,
            },
        }
        if error:
            item["error"] = {"type": type(error).__name__, "message": str(error)}
            item["error_classification"] = classify_tool_error(error)
        else:
            item["result"] = _trace_result(result)
            # Keep the actual return paths/types available for a later repair;
            # values themselves remain omitted from the trace.
            item["result_schema"] = describe_value(result)
            reusable = _cacheable_tool_result(result)
            if reusable is not None:
                self._reusable_results[item["cache_key"]] = reusable
        self.calls.append(item)
        self.ledger.record(name, arguments, result, execution_status)


class ToolInterface:
    """Transparent public-method proxy used only for call observability."""

    def __init__(
        self,
        name_or_component: str | Any,
        component_or_methods: Any | tuple[str, ...],
        allowed_methods: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(name_or_component, str):
            self._name = str(name_or_component)
            self._component = component_or_methods
        else:
            self._name = "tool"
            self._component = name_or_component

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(self._component, name)
        if not callable(method):
            return method

        def invoke(*args: Any, **kwargs: Any) -> Any:
            return method(*args, **kwargs)

        invoke.__name__ = getattr(method, "__name__", name)
        invoke.__doc__ = getattr(method, "__doc__", None)
        try:
            invoke.__signature__ = inspect.signature(method)
        except (TypeError, ValueError):
            pass

        return invoke

    def __dir__(self) -> list[str]:
        public = [
            name for name in dir(self._component)
            if not name.startswith("_") and callable(getattr(self._component, name, None))
        ]
        return sorted(set(super().__dir__()) | set(public))


class LazyToolInterface:
    """Load one capability only when generated code first accesses it."""

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        self._name = str(name)
        self._loader = loader
        self._interface: ToolInterface | None = None

    @property
    def loaded(self) -> bool:
        return self._interface is not None

    def _load(self) -> ToolInterface:
        if self._interface is None:
            self._interface = ToolInterface(self._name, self._loader())
        return self._interface

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


class DatasetRecord(dict):
    """Data record with collision-safe access for documented data fields.

    ``dict.values`` is a method, so a normal ``dict`` subclass cannot safely
    expose a data field named ``values`` through attribute syntax.  Preserve
    bracket access for all keys and make only data-field attributes explicit.
    """

    _DATA_ATTRIBUTES = frozenset({"values", "raw_data", "dimensions", "relative_minutes", "metadata"})

    def __getattribute__(self, name: str) -> Any:
        if name in DatasetRecord._DATA_ATTRIBUTES:
            try:
                return dict.__getitem__(self, name)
            except KeyError:
                pass
        return dict.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class DataIndexer:
    """Server-side loader that resolves data_descs before generated code runs."""

    @staticmethod
    def _raw_upload(upload_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Load the upload's primitive pixel arrays into ``ds``.

        This is deliberately a data-loading operation, not an analysis API.
        No threshold, field registry, region detector, or summary statistic is
        applied here.  The generated program receives the pixels and chooses
        its own fields and calculations, matching Zephyrus's xarray ``ds``
        contract as closely as the JSON worker boundary allows.
        """
        kind = str(metadata.get("kind", "single_frame"))
        if kind == "sequence":
            sequence, loaded_metadata = _load_uploaded_vil_sequence(upload_id)
            # Preserve every encoded value exactly while exposing float32 for
            # arithmetic. Direct subtraction on uint8 silently wraps at 0/255
            # and previously produced false changes such as +77.75 instead of
            # +14.86. The original storage dtype remains explicit provenance.
            values = np.ascontiguousarray(sequence, dtype=np.float32)
            values.setflags(write=False)
            return {
                "kind": "sequence",
                "values": values,
                "relative_minutes": [
                    int(value) for value in loaded_metadata.get("history_minutes", [])
                ],
                "shape": [int(value) for value in sequence.shape],
                "dtype": "float32",
                "source_dtype": "uint8",
                "missing_value_code": 255,
                "raw_data": values,
                "statistics_computed": [],
                "fields_computed": [],
            }
        frame, _ = _load_uploaded_vil(upload_id)
        values = np.ascontiguousarray(frame, dtype=np.float32)
        values.setflags(write=False)
        return {
            "kind": "single_frame",
            "values": values,
            "shape": [int(value) for value in frame.shape],
            "dtype": "float32",
            "source_dtype": "uint8",
            "missing_value_code": 255,
            "raw_data": values,
            "statistics_computed": [],
            "fields_computed": [],
        }

    def load(self, data_descs: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for description in data_descs:
            source = str(description.get("source", "sevir")).lower()
            if source == "sevir":
                from ..data_access.catalog import (
                    get_sevir_event_metadata,
                    load_sevir_vil_event,
                )
                event_id = str(description.get("event_id", "")).strip()
                if not event_id:
                    raise ValueError("SEVIR data_desc requires event_id")
                metadata = normalize_tool_result(
                    "sevir.event", get_sevir_event_metadata(event_id)
                )
                event_metadata = metadata["record"]
                if str(description.get("representation", "")).lower() == "observation_sequence":
                    loaded = normalize_tool_result(
                        "sevir.observations", load_sevir_vil_event(event_id, 0, 48)
                    )
                    if str(loaded.get("status", "")) != "ok":
                        raise ValueError(
                            f"SEVIR observation sequence unavailable for {event_id}: "
                            f"{loaded.get('error_code') or loaded.get('status') or 'unknown'}"
                        )
                    values = loaded.get("frames")
                    if values is None:
                        values = loaded.get("data", {}).get("values")
                    relative_minutes = list(
                        loaded.get("relative_minutes")
                        or loaded.get("data", {}).get("relative_minutes")
                        or []
                    )
                    if not isinstance(values, np.ndarray) or values.ndim != 3:
                        raise ValueError(f"SEVIR observation sequence has invalid shape for {event_id}")
                    window = description.get("observation_window_minutes")
                    if isinstance(window, (list, tuple)) and len(window) == 2 and relative_minutes:
                        lower, upper = int(window[0]), int(window[1])
                        indices = [
                            index for index, minute in enumerate(relative_minutes)
                            if lower <= int(minute) <= upper
                        ]
                        if not indices:
                            raise ValueError(
                                f"SEVIR observation window {lower}..{upper} is empty for {event_id}"
                            )
                        values = values[indices[0]:indices[-1] + 1]
                        relative_minutes = relative_minutes[indices[0]:indices[-1] + 1]
                    values = np.ascontiguousarray(values)
                    values.setflags(write=False)
                    datasets.append(DatasetRecord({
                        "source": "sevir",
                        "role": "local_observation",
                        "dataset_id": f"sevir:{event_id}:observations",
                        "event_id": event_id,
                        "kind": "sequence",
                        "status": "available",
                        "capabilities": [
                            "encoded_vil", "local_observation", "temporal_sequence",
                            "event_metadata", "event_time", "event_location",
                        ],
                        "metadata": event_metadata,
                        "shape": [int(item) for item in values.shape],
                        "dimensions": ["time", "row", "column"],
                        "dtype": str(values.dtype),
                        "source_dtype": str(values.dtype),
                        "missing_value_code": int(loaded.get("missing_value", 255)),
                        "relative_minutes": [int(item) for item in relative_minutes],
                        "raw_data": values,
                        "values": values,
                        "statistics_computed": [],
                        "fields_computed": [],
                    }))
                else:
                    datasets.append(
                        DatasetRecord({
                            "source": "sevir",
                            "role": "event_metadata",
                            "dataset_id": f"sevir:{event_id}",
                            "event_id": event_id,
                            "status": "available",
                            "capabilities": ["event_metadata", "event_time", "event_location"],
                            "metadata": event_metadata,
                        })
                    )
            elif source == "foundation_case":
                from ..data_access.foundation import get_case_foundation_artifacts
                from ..data_access.foundation_validation import build_foundation_bundle
                event_id = str(description.get("event_id", "")).strip()
                if not event_id:
                    raise ValueError("foundation_case data_desc requires event_id")
                registry = get_case_foundation_artifacts(event_id)
                evidence_path = ROOT / "data" / "era5_cases" / event_id / "artifacts" / "case_evidence.json"
                evidence = (
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                    if evidence_path.is_file()
                    else registry.get("evidence")
                )
                bundle = build_foundation_bundle(event_id, evidence, registry.get("artifacts", []))
                datasets.append(DatasetRecord({
                    "source": "foundation_case",
                    "role": "environmental_evidence",
                    "dataset_id": f"foundation:{event_id}",
                    "event_id": event_id,
                    "selection": str(description.get("selection", "explicit_case")),
                    "evidence": evidence,
                    "evidence_bundle": bundle,
                    "validation": bundle["validation"],
                    "artifacts": registry.get("artifacts", []),
                    "status": registry.get("status", "not_found"),
                    "capabilities": ["era5_reanalysis", "aurora_forecast", "verification"] if registry.get("status") == "ok" else [],
                    "interpretation_limit": (
                        "SEVIR/CasCast local VIL, ERA5 retrospective environment and Aurora "
                        "large-scale forecast are separate evidence roles."
                    ),
                }))
            elif source == "upload":
                from ..data_access.uploads import list_uploaded_records
                upload_id = str(description.get("upload_id", "")).strip()
                records = list_uploaded_records().get("records", [])
                metadata = next((item for item in records if item.get("upload_id") == upload_id), None)
                if metadata is None:
                    raise ValueError(f"unknown upload_id: {upload_id}")
                # Upload records contain convenience summaries for the UI.
                # Do not expose those summaries as analysis inputs: ds carries
                # only the upload metadata needed for provenance plus raw data.
                analysis_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"min_vil", "max_vil", "mean_vil"}
                }
                primitive = self._raw_upload(upload_id, analysis_metadata)
                datasets.append(
                    DatasetRecord({
                        "source": "upload",
                        "role": "local_observation",
                        "dataset_id": f"upload:{upload_id}",
                        "upload_id": upload_id,
                        "metadata": analysis_metadata,
                        "kind": primitive["kind"],
                        "shape": primitive["shape"],
                        "dimensions": ["time", "row", "column"] if primitive["kind"] == "sequence" else ["row", "column"],
                        "dtype": primitive["dtype"],
                        "source_dtype": primitive.get("source_dtype", primitive["dtype"]),
                        "missing_value_code": primitive.get("missing_value_code", 255),
                        "relative_minutes": primitive.get("relative_minutes", []),
                        "status": "available",
                        "capabilities": ["encoded_vil", "local_observation"] + (["temporal_sequence"] if primitive["kind"] == "sequence" else []),
                        # Keep the generated-program contract simple: raw_data
                        # is the actual 2-D/3-D uint8 pixel list, not another
                        # wrapper object and not a boolean marker.
                        "raw_data": primitive["raw_data"],
                        "values": primitive["values"],
                        "statistics_computed": [],
                        "fields_computed": [],
                    })
                )
            else:
                raise ValueError(f"unsupported data source: {source}")
        if datasets:
            for index, item in enumerate(datasets):
                item["is_primary"] = index == 0
        return datasets


def describe_request_data(data_descs: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Load only metadata/raw-shape information for prompt construction.

    This mirrors Zephyrus's dataset_describe endpoint.  It is best-effort so a
    missing optional HDF5 asset does not prevent the Agent from receiving the
    static contract and producing a useful repairable execution.
    """
    try:
        loaded = DataIndexer().load(tuple(dict(item) for item in data_descs))
        return describe_dataset(loaded)
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "object": "ds",
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "requested_data_descs": [dict(item) for item in data_descs],
        }


class ExecutionEnvironment:
    """Assemble request-scoped resources in the canonical Zephyrus order."""

    def __init__(
        self,
        request: ExecutionRequest,
        replay_results: dict[str, Any] | None = None,
    ) -> None:
        self.request = request
        self.recorder = ToolUsageRecorder(replay_results)
        requested_names = {
            canonical_tool_name(name)
            for name in request.selected_tools
            if name not in {"ds"}
        }
        self.resources: dict[str, Any] = {"ds": DataIndexer().load(request.data_descs)}
        default_event_id = next(
            (
                str(item.get("event_id", "")).strip().upper()
                for item in request.data_descs
                if item.get("source") in {"sevir", "foundation_case"}
                and item.get("event_id")
            ),
            "",
        )
        default_upload_id = next(
            (
                str(item.get("upload_id", "")).strip()
                for item in request.data_descs
                if item.get("source") == "upload" and item.get("upload_id")
            ),
            "",
        )

        def load_capability(name: str) -> Any:
            tool_class = _load_tool_class(name)
            return tool_class(
                self.recorder,
                default_event_id=default_event_id,
                default_upload_id=default_upload_id,
            )

        for name in CANONICAL_ARGUMENTS:
            if name == "ds" or name not in requested_names:
                continue
            self.resources[name] = LazyToolInterface(
                name,
                lambda selected=name: load_capability(selected),
            )

    def arguments(self) -> list[Any]:
        return [self.resources[name] for name in CANONICAL_ARGUMENTS if name in self.request.selected_tools]

    def describe(self) -> dict[str, Any]:
        """Expose the actual request-scoped data shape and selected tool contract."""
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": describe_dataset(self.resources["ds"]),
            "tools": [
                {
                    "name": name,
                    "loading": "lazy",
                    "loaded": bool(getattr(self.resources[name], "loaded", False)),
                }
                for name in self.request.selected_tools
                if name != "ds"
            ],
        }

    def invoke(self, run_function: Callable[..., Any]) -> Any:
        return run_function(*self.arguments())
