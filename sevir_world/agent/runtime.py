"""Assemble request-scoped execution resources."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import scipy
import skimage

from .contracts import ExecutionRequest, WORKER_CONTRACT_VERSION
from .study import StudyEvidence
from .case_memory import CASE_MEMORY_SEARCH_SCHEMA_VERSION, CaseMemory, CaseMemoryStore
from ..capabilities.base import ToolUsageRecorder, ToolInterface, LazyToolInterface
from ..capabilities.registry import get_capability, load_capability
from ..serialization import describe_value
from ..tools.analytics.public import Analytics


SCHEMA_VERSION = "python-capabilities-v1"


class DatasetRecord(dict):
    """Keep the documented attribute access for data keys that shadow dict methods."""
    _DATA_ATTRIBUTES = frozenset({"values", "raw_data", "dimensions", "relative_minutes", "metadata"})

    def __getattribute__(self, name: str) -> Any:
        if name in DatasetRecord._DATA_ATTRIBUTES and name in self:
            return self[name]
        return dict.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self:
            raise AttributeError(name)
        return self[name]


class DatasetCollection(Sequence):
    """Load and retain each request-bound resource only when it is accessed."""

    def __init__(self, descriptions: tuple[dict[str, Any], ...]) -> None:
        self.descriptions = descriptions
        self._loaded: dict[int, DatasetRecord] = {}

    def __len__(self) -> int:
        return len(self.descriptions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        description = self.descriptions[index]
        index = index if index >= 0 else len(self) + index
        if index not in self._loaded:
            if description["source"] == "sevir":
                from ..tools.sevir.repository import load_request_dataset as loader
            else:
                from ..tools.sevir.uploads import load_uploaded_dataset as loader
            self._loaded[index] = DatasetRecord(loader(description), is_primary=index == 0)
        from ..execution_state import isolated_copy
        return isolated_copy(self._loaded[index])

    def peek(self, index: int) -> DatasetRecord | None:
        """Inspect an already loaded record without triggering data access."""
        return self._loaded.get(index)

    def describe(self) -> dict[str, Any]:
        """Describe request-bound data without loading or copying raster values."""
        records = []
        for index, description in enumerate(self.descriptions):
            loaded = self.peek(index)
            item = loaded if loaded is not None else description
            records.append({
                "index": index,
                "loading": "loaded" if loaded is not None else "lazy",
                "source": item.get("source"),
                "role": item.get("role"),
                "event_id": item.get("event_id"),
                "upload_id": item.get("upload_id"),
                "shape": item.get("shape"),
                "fields": {
                    key: describe_value(item[key])
                    for key in ("values", "raw_data", "relative_minutes", "dimensions", "metadata")
                    if key in item
                },
            })
        return {"schema_version": SCHEMA_VERSION, "object": "ds", "records": records}


class ExecutionEnvironment:
    """Build the one resource namespace used by generated Python code."""

    available_modules = ("numpy", "pandas", "xarray", "matplotlib", "scipy", "skimage")

    def __init__(self, request: ExecutionRequest) -> None:
        self.request = request
        self.recorder = ToolUsageRecorder()
        self.resources: dict[str, Any] = {
            "ds": DatasetCollection(request.data_descs),
            "study": StudyEvidence(request.evidence_state),
            "case_memory": CaseMemory(
                CaseMemoryStore(), current_case_id=str(request.evidence_state.get("case_id") or ""),
            ),
            "analytics": ToolInterface("analytics", Analytics(self.recorder), self.recorder),
            "np": np, "pd": pd, "xr": xr, "scipy": scipy, "skimage": skimage, "math": math,
        }
        if request.generate_visual:
            from ..tools.visualization.renderer import plot_field
            from ..tools.visualization.styles import matplotlib_style

            self.resources.update(plot_field=plot_field, plot_style=matplotlib_style)
        for name in request.selected_tools:
            if name != "ds":
                self.resources[name] = LazyToolInterface(
                    name, lambda selected=name: load_capability(selected, self.recorder, request.binding),
                    self.recorder,
                )

    def describe(self) -> dict[str, Any]:
        """Snapshot assembled resources without loading datasets or lazy tools."""
        return {
            "schema_version": SCHEMA_VERSION,
            "worker_contract_version": WORKER_CONTRACT_VERSION,
            "available_modules": self.available_modules,
            "dataset": self.resources["ds"].describe(),
            "study": {
                "case_id": self.resources["study"].case_id,
                "revision": self.resources["study"].revision,
            },
            "case_memory": {
                "schema_version": CASE_MEMORY_SEARCH_SCHEMA_VERSION,
                "mode": "explicit_cross_session_evidence_search",
                "answer_prose_indexed": False,
                "implicit_inheritance": False,
            },
            "analytics": Analytics._contract_snapshot(),
            "generate_visual": self.request.generate_visual,
            "tools": [
                {
                    "name": name,
                    "display_name": get_capability(name).display_name,
                    "loading": "lazy",
                    "loaded": self.resources[name].loaded,
                }
                for name in self.request.selected_tools if name != "ds"
            ],
        }
