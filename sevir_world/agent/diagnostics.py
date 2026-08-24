from __future__ import annotations

"""Small runtime diagnostics; these describe values but do not define tools."""

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


SCHEMA_VERSION = "python-capabilities-v1"


def classify_execution_error(error_type: str, message: str) -> dict[str, Any]:
    """Give the repair loop one generic traceback-driven instruction."""

    return {
        "category": str(error_type or "ExecutionError"),
        "action": "use the traceback and the documented method signature to rewrite the failing code",
        "message": str(message)[:2000],
    }


def describe_value(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 5:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray", "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype), "read_only": not bool(value.flags.writeable),
        }
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


def describe_dataset(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe request-bound data without copying raster values."""

    records = []
    for index, item in enumerate(data):
        fields_description = {
            key: describe_value(item[key])
            for key in ("values", "raw_data", "relative_minutes", "dimensions", "metadata")
            if key in item
        }
        records.append({
            "index": index,
            "source": item.get("source"),
            "role": item.get("role"),
            "event_id": item.get("event_id"),
            "upload_id": item.get("upload_id"),
            "shape": item.get("shape"),
            "fields": fields_description,
        })
    return {"schema_version": SCHEMA_VERSION, "object": "ds", "records": records}
