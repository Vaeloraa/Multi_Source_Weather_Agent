"""Shared scientific-value descriptions and conversion to JSON-compatible values."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .capabilities.dataframe_schema import dataframe_schema_metadata


@dataclass(frozen=True)
class SummaryLimits:
    """Limits selected by the caller for a lossy, human-readable summary."""

    max_characters: int
    max_depth: int = 7
    max_string_characters: int = 2_000
    max_mapping_items: int = 80
    max_sequence_items: int = 16
    dataframe_preview_rows: int = 20


def serialize_value(
    value: Any, *, inline_array_elements: int = 0,
    summary: SummaryLimits | None = None, depth: int = 0,
) -> Any:
    """Convert values using one traversal for transport and bounded summaries.

    Without ``summary``, preserve supported container contents and reject
    unsupported objects or excessive nesting. Scientific objects stay local,
    except arrays within the caller's inline limit. With ``summary``, truncate
    oversized contents and describe unsupported objects by type.
    """

    def convert(item: Any, level: int) -> Any:
        if level > (summary.max_depth if summary else 40):
            if summary is None:
                raise ValueError("result nesting exceeds 40 levels")
            return {"type": type(item).__name__, "trace_truncated": True}
        if isinstance(item, str):
            if summary and len(item) > summary.max_string_characters:
                return {
                    "type": "str", "length": len(item),
                    "preview": item[:summary.max_string_characters], "trace_truncated": True,
                }
            return item
        if item is None or isinstance(item, (int, float, bool)):
            return None if isinstance(item, float) and not math.isfinite(item) else item
        if isinstance(item, np.generic):
            return convert(item.item(), level + 1)
        if summary is None and isinstance(item, np.ndarray) and item.size <= inline_array_elements:
            return convert(item.tolist(), level + 1)
        metadata = value_metadata(item)
        if metadata is not None:
            if summary is not None:
                if isinstance(item, pd.DataFrame):
                    rows = item.head(summary.dataframe_preview_rows).to_dict(orient="records")
                    metadata.update(
                        records=[{str(key): convert(part, level + 1) for key, part in row.items()}
                                 for row in rows],
                        records_returned=len(rows),
                    )
                return {**metadata, "trace_values_omitted": True}
            omitted = "array_values_omitted" if isinstance(item, np.ndarray) else (
                "rows_omitted" if isinstance(item, pd.DataFrame) else "values_omitted"
            )
            return convert({**metadata, omitted: True}, level + 1)
        if is_dataclass(item) and not isinstance(item, type):
            converted = {field.name: convert(getattr(item, field.name), level + 1) for field in fields(item)}
            return {"type": type(item).__name__, **converted} if summary else converted
        if isinstance(item, Mapping):
            pairs = islice(item.items(), summary.max_mapping_items) if summary else item.items()
            converted = {str(key): convert(part, level + 1) for key, part in pairs}
            if summary and len(item) > summary.max_mapping_items:
                converted["trace_truncated"] = True
            return converted
        if isinstance(item, (list, tuple)) or (summary is None and isinstance(item, set)):
            if summary and len(item) > summary.max_sequence_items:
                return {
                    "type": type(item).__name__, "count": len(item),
                    "first": convert(item[0], level + 1), "last": convert(item[-1], level + 1),
                }
            return [convert(part, level + 1) for part in item]
        if isinstance(item, Path):
            return str(item)
        if summary:
            return {"type": type(item).__name__}
        raise TypeError(f"result contains unsupported type: {type(item).__name__}")

    converted = convert(value, depth)
    if summary:
        size = len(json.dumps(converted, ensure_ascii=False))
        if size > summary.max_characters:
            return {"type": type(value).__name__, "trace_truncated": True, "original_json_characters": size}
    return converted


def value_metadata(value: Any) -> dict[str, Any] | None:
    """Describe scientific objects without reading their array values."""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype),
            "elements": int(value.size), "read_only": not value.flags.writeable,
        }
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame", "representation": "metadata_only", "read_scope": "worker_only",
            "shape": list(value.shape), "columns": [str(name) for name in value.columns],
            **dataframe_schema_metadata(value),
            **{name: value.attrs[name] for name in (
                "criteria", "total_matches", "returned_count", "truncated", "availability", "units",
            ) if name in value.attrs},
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series", "representation": "metadata_only", "read_scope": "worker_only",
            "length": len(value), "dtype": str(value.dtype), "name": str(value.name or ""),
        }
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        result = {
            "type": type(value).__name__, "representation": "metadata_only", "read_scope": "worker_only",
            "coordinates": [str(name) for name in value.coords],
        }
        if isinstance(value, xr.Dataset):
            result.update(
                dimensions={str(name): int(size) for name, size in value.sizes.items()},
                variables=list(value.data_vars),
                variable_metadata={
                    str(name): {"dimensions": list(field.dims), "dtype": str(field.dtype),
                                **({"units": field.attrs["units"]} if "units" in field.attrs else {})}
                    for name, field in islice(value.data_vars.items(), 40)
                },
            )
        else:
            result.update(name=str(value.name or ""), dimensions=list(value.dims),
                          shape=list(value.shape), dtype=str(value.dtype))
            if "units" in value.attrs:
                result["units"] = value.attrs["units"]
        result["provenance_metadata"] = {
            str(name): item if not isinstance(item, tuple) else list(item)
            for name, item in value.attrs.items()
            if isinstance(item, (str, int, float, bool)) or (
                isinstance(item, (list, tuple)) and len(item) <= 48
                and all(part is None or isinstance(part, (str, int, float, bool)) for part in item)
            )
        }
        for name in ("time_selection", "spatial_selection", "availability"):
            if name in value.attrs:
                result["provenance_metadata"][name] = value.attrs[name]
        return result
    return None


def describe_value(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 5:
        return {"type": type(value).__name__, "truncated": True}
    metadata = value_metadata(value)
    if metadata is not None:
        return metadata
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            "fields": {field.name: describe_value(getattr(value, field.name), depth + 1) for field in fields(value)},
        }
    if isinstance(value, dict):
        return {"type": "dict", "fields": {str(key): describe_value(item, depth + 1) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__, "length": len(value),
            "items": describe_value(value[0], depth + 1) if value else None,
        }
    return {"type": type(value).__name__}
