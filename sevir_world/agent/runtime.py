from __future__ import annotations

import json
import importlib
import inspect
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr
import scipy
import skimage

from .contracts import CANONICAL_ARGUMENTS, ExecutionRequest
from .diagnostics import (
    SCHEMA_VERSION,
    describe_dataset,
)
from ..data_access.uploads import _load_uploaded_vil, _load_uploaded_vil_sequence
from .study import StudyEvidence


_TOOL_IMPORTS = {
    "sevir": ("sevir_world.tools.sevir.Sevir", "Sevir"),
    "geolocator": ("sevir_world.tools.geolocator.Geolocator", "Geolocator"),
    "cascast": ("sevir_world.tools.cascast.CasCast", "CasCast"),
    "aurora": ("sevir_world.tools.aurora.Aurora", "Aurora"),
}

_LOADED_TOOL_CLASSES: dict[str, type[Any]] = {}

AVAILABLE_MODULES = ("numpy", "pandas", "xarray", "matplotlib", "scipy", "skimage")


def _load_tool_class(name: str) -> type[Any]:
    """Import one concrete tool only when the bound request needs it."""

    canonical = str(name)
    if canonical not in _TOOL_IMPORTS:
        raise RuntimeError(f"unknown worker tool: {name}")
    loaded = _LOADED_TOOL_CLASSES.get(canonical)
    if loaded is None:
        module_name, class_name = _TOOL_IMPORTS[canonical]
        loaded = getattr(importlib.import_module(module_name), class_name)
        _LOADED_TOOL_CLASSES[canonical] = loaded
    return loaded


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
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "representation": "metadata_only",
                "read_scope": "worker_only",
                "shape": [int(item) for item in value.shape],
                "columns": [str(item) for item in value.columns],
                "records": [
                    {str(key): array_metadata(item) for key, item in row.items()}
                    for row in value.head(20).to_dict(orient="records")
                ],
                "records_returned": min(20, int(len(value))),
                "trace_values_omitted": True,
            }
        if type(value).__module__.startswith("xarray"):
            data_vars = getattr(value, "data_vars", None)
            if data_vars is not None:
                return {
                    "type": type(value).__name__,
                    "representation": "metadata_only",
                    "read_scope": "worker_only",
                    "dimensions": {str(key): int(size) for key, size in value.sizes.items()},
                    "variables": [str(name) for name in data_vars],
                    "coordinates": [str(name) for name in value.coords],
                    "trace_values_omitted": True,
                }
            return {
                "type": type(value).__name__,
                "representation": "metadata_only",
                "read_scope": "worker_only",
                "name": str(getattr(value, "name", "") or ""),
                "dimensions": [str(name) for name in getattr(value, "dims", ())],
                "shape": [int(size) for size in getattr(value, "shape", ())],
                "dtype": str(getattr(value, "dtype", "")),
                "trace_values_omitted": True,
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": type(value).__name__,
                **{
                    field.name: array_metadata(getattr(value, field.name))
                    for field in fields(value)
                },
            }
        if isinstance(value, dict):
            return {str(key): array_metadata(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)) and len(value) > 16:
            return {
                "type": type(value).__name__,
                "count": len(value),
                "first": array_metadata(value[0]),
                "last": array_metadata(value[-1]),
            }
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


class ToolCostBudgetExceeded(RuntimeError):
    """Raised before a backend call would exceed the request cost budget."""


class ToolUsageRecorder:
    """Record calls and enforce a persistent weighted runtime cost budget."""

    def __init__(self, cost_budget_units: float | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        configured = (
            float(cost_budget_units)
            if cost_budget_units is not None
            else float(os.getenv("SEVIR_AGENT_TOOL_COST_BUDGET", "30"))
        )
        if configured <= 0:
            raise ValueError("tool cost budget must be positive")
        self.cost_budget_units = configured
        self.cost_used_units = 0.0

    @staticmethod
    def estimate_cost(name: str, arguments: dict[str, Any], *, cache_hit: bool = False) -> float:
        """Estimate backend work in stable abstract units, independent of Python syntax."""

        base = {
            "sevir.event": 1.0,
            "sevir.search": 1.5,
            "sevir.observations": 3.0,
            "sevir.match_upload": 1.0,
            "cascast.forecast": 7.0,
            "aurora.regional_fields": 8.0,
        }.get(name, 1.0)
        if name == "sevir.observation_batch":
            count = len(arguments.get("subject_keys") or ())
            base = 2.0 + min(max(count, 1), 20) * 0.75
        if cache_hit:
            base = min(base, 0.25)
        return round(float(base), 2)

    def reserve(
        self, name: str, arguments: dict[str, Any], *, cache_hit: bool = False
    ) -> float:
        """Reserve cost before backend execution so loops cannot evade the budget."""

        cost = self.estimate_cost(name, arguments, cache_hit=cache_hit)
        projected = self.cost_used_units + cost
        if projected > self.cost_budget_units + 1e-9:
            raise ToolCostBudgetExceeded(
                f"{name} costs {cost:g} units but only "
                f"{self.remaining_cost_units:g} of {self.cost_budget_units:g} remain"
            )
        self.cost_used_units = round(projected, 2)
        return cost

    @property
    def remaining_cost_units(self) -> float:
        return round(max(0.0, self.cost_budget_units - self.cost_used_units), 2)

    def summary(self) -> dict[str, float | bool]:
        return {
            "budget_units": self.cost_budget_units,
            "used_units": self.cost_used_units,
            "remaining_units": self.remaining_cost_units,
            "exhausted": self.remaining_cost_units <= 0,
        }

    def __call__(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
        cache_hit: bool = False,
        cost_units: float | None = None,
    ) -> None:
        execution_status = "error" if error else "ok"
        item: dict[str, Any] = {
            "step": len(self.calls) + 1,
            "api": name,
            "arguments": _trace_arguments(arguments),
            "status": execution_status,
            "cache_hit": bool(cache_hit),
            "cost_units": float(cost_units or 0.0),
            "cumulative_cost_units": self.cost_used_units,
            "remaining_cost_units": self.remaining_cost_units,
        }
        if error:
            item["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            item["result"] = _trace_result(result)
            item["return_type"] = type(result).__name__
        self.calls.append(item)


class ToolInterface:
    """Transparent public-method proxy used only for call observability."""

    def __init__(
        self,
        name: str,
        component: Any,
    ) -> None:
        self._name = str(name)
        self._component = component

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
                from ..tools.sevir.Catalog import (
                    get_sevir_event_metadata,
                    load_sevir_vil_event,
                )
                event_id = str(description.get("event_id", "")).strip()
                if not event_id:
                    raise ValueError("SEVIR data_desc requires event_id")
                metadata = get_sevir_event_metadata(event_id)
                if metadata.get("status") != "ok":
                    raise ValueError(f"SEVIR metadata unavailable for {event_id}")
                event_metadata = metadata["record"]
                if str(description.get("representation", "")).lower() == "observation_sequence":
                    loaded = load_sevir_vil_event(event_id, 0, 48)
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


def _plot_field(
    data: Any,
    title: str = "",
    cmap: str = "viridis",
    filename: str | None = None,
) -> str:
    """Save one quick evidence figure and return the PNG path.

    Pre-imported into the execution namespace as ``plot_field`` so generated
    programs can produce the mandatory figure with a single call instead of
    spending hundreds of output tokens on matplotlib boilerplate.  A 3-D cube
    plots its last frame.
    """

    import os
    import uuid

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    array = np.asarray(data)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        raise ValueError("plot_field expects a 2-D field or a 3-D cube")
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap=cmap, origin="upper")
    figure.colorbar(image, ax=axis)
    if title:
        axis.set_title(str(title))
    artifact_dir = os.environ.get("SEVIR_ARTIFACT_DIR", "")
    if not artifact_dir:
        raise RuntimeError("SEVIR_ARTIFACT_DIR is not configured")
    name = str(filename).strip() if filename else f"{uuid.uuid4().hex}.png"
    if not name.lower().endswith(".png"):
        name += ".png"
    path = Path(artifact_dir) / name
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return str(path)


class ExecutionEnvironment:
    """Assemble request-scoped resources in the canonical Zephyrus order."""

    def __init__(
        self,
        request: ExecutionRequest,
    ) -> None:
        self.request = request
        self.recorder = ToolUsageRecorder()
        self.resources: dict[str, Any] = {
            "ds": DataIndexer().load(request.data_descs),
            "study": StudyEvidence(request.evidence_state),
            "plot_field": _plot_field,
            "pd": pd,
            "xr": xr,
            "scipy": scipy,
            "skimage": skimage,
        }
        def load_capability(name: str) -> Any:
            tool_class = _load_tool_class(name)
            return tool_class(self.recorder, request.binding)

        # Runtime availability is independent of prompt/document selection.
        # Every request receives the same four lazy capability objects; no
        # backend is imported until generated code first accesses that object.
        # ``selected_tools`` only determines which contracts are expanded in
        # the prompt, so a routing hint can shorten context without removing
        # an Agent capability from the persistent namespace.
        for name in CANONICAL_ARGUMENTS:
            if name == "ds":
                continue
            self.resources[name] = LazyToolInterface(
                name,
                lambda selected=name: load_capability(selected),
            )

    def describe(self) -> dict[str, Any]:
        """Expose the actual request-scoped data shape and selected tool contract."""
        return {
            "schema_version": SCHEMA_VERSION,
            "available_modules": AVAILABLE_MODULES,
            "dataset": describe_dataset(self.resources["ds"]),
            "study": {
                "case_id": self.resources["study"].case_id,
                "revision": self.resources["study"].revision,
            },
            "tools": [
                {
                    "name": name,
                    "loading": "lazy",
                    "loaded": bool(getattr(self.resources[name], "loaded", False)),
                }
                for name in CANONICAL_ARGUMENTS
                if name != "ds"
            ],
        }
