from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_ARGUMENTS = (
    "ds",
    "geolocator",
    "forecaster",
    "statistics",
    "uncertainty",
    "metrics",
    "visualizer",
)
DEFAULT_SELECTED_TOOLS = CANONICAL_ARGUMENTS


@dataclass(frozen=True)
class ExecutionRequest:
    """Normalized input contract shared by Direct and future Reflective modes."""

    data_descs: tuple[dict[str, Any], ...]
    selected_tools: tuple[str, ...]

    @classmethod
    def from_payload(
        cls,
        data_descs: list[dict[str, Any]] | None,
        selected_tools: list[str] | tuple[str, ...] | None,
    ) -> "ExecutionRequest":
        descriptions = tuple(dict(item) for item in (data_descs or []))
        requested = tuple(selected_tools or DEFAULT_SELECTED_TOOLS)
        unknown = sorted(set(requested) - set(CANONICAL_ARGUMENTS))
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        if len(requested) != len(set(requested)):
            raise ValueError("selected_tools contains duplicates")
        ordered = tuple(name for name in CANONICAL_ARGUMENTS if name in requested)
        if "ds" not in ordered:
            raise ValueError("selected_tools must include ds")
        return cls(descriptions, ordered)


def data_descs_from_context(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build the legacy HTTP context's explicit dataset description."""
    bound = dict(context or {})
    if bound.get("data_descs"):
        return [dict(item) for item in bound["data_descs"]]
    if bound.get("upload_id"):
        return [{"source": "upload", "upload_id": str(bound["upload_id"])}]
    if bound.get("sequence_upload_id"):
        return [{"source": "upload", "upload_id": str(bound["sequence_upload_id"])}]
    if bound.get("event_id"):
        return [{"source": "sevir", "event_id": str(bound["event_id"])}]
    return []
