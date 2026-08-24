from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_ARGUMENTS = (
    "ds",
    "sevir",
    "geolocator",
    "cascast",
    "aurora",
)
DEFAULT_SELECTED_TOOLS = CANONICAL_ARGUMENTS
DATA_CONTRACT_VERSION = "sevir-data-v3"
_DATA_SOURCES = {"upload", "sevir", "foundation_case"}


@dataclass(frozen=True)
class ExecutionRequest:
    """Data identities and coarse tool objects injected into one namespace."""

    data_descs: tuple[dict[str, Any], ...]
    selected_tools: tuple[str, ...]

    @classmethod
    def from_payload(
        cls,
        data_descs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        selected_tools: list[str] | tuple[str, ...] | None,
    ) -> "ExecutionRequest":
        raw = normalize_data_descs(data_descs)
        priority = {"upload": 0, "sevir": 1, "foundation_case": 2}
        descriptions = tuple(
            sorted(
                raw,
                key=lambda item: priority.get(
                    str(item.get("source", "")).lower(), 9
                ),
            )
        )
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


def normalize_data_descs(
    data_descs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> list[dict[str, Any]]:
    if data_descs is None:
        return []
    if not isinstance(data_descs, (list, tuple)):
        raise ValueError("data_descs must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(data_descs):
        if not isinstance(raw, dict):
            raise ValueError(f"data_descs[{index}] must be an object")
        item = dict(raw)
        source = str(item.get("source", "")).strip().lower()
        if source not in _DATA_SOURCES:
            raise ValueError(
                f"data_descs[{index}].source must be one of "
                "upload, sevir, foundation_case"
            )
        item["source"] = source
        key_name = "upload_id" if source == "upload" else "event_id"
        identity_value = str(item.get(key_name, "")).strip()
        if not identity_value:
            raise ValueError(f"data_descs[{index}] requires {key_name}")
        if source != "upload":
            identity_value = identity_value.upper()
        item[key_name] = identity_value
        identity = (source, identity_value)
        if identity in seen:
            continue
        seen.add(identity)
        item.setdefault("contract_version", DATA_CONTRACT_VERSION)
        normalized.append(item)
    return normalized


def data_descs_from_context(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    bound = dict(context or {})
    if bound.get("data_descs"):
        return normalize_data_descs(bound["data_descs"])
    if bound.get("sequence_upload_id"):
        return normalize_data_descs([
            {"source": "upload", "upload_id": str(bound["sequence_upload_id"])}
        ])
    if bound.get("upload_id"):
        return normalize_data_descs([
            {"source": "upload", "upload_id": str(bound["upload_id"])}
        ])
    if bound.get("event_id"):
        return normalize_data_descs([
            {"source": "sevir", "event_id": str(bound["event_id"])}
        ])
    return []


def data_descs_for_event(
    event_id: str,
    descriptions: list[dict[str, Any]],
    *,
    include_observations: bool = False,
    observation_window_minutes: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Bind an explicit event while preserving uploaded data as primary."""

    normalized_id = str(event_id or "").strip().upper()
    if not normalized_id:
        return [dict(item) for item in descriptions]
    existing = next(
        (
            dict(item) for item in descriptions
            if item.get("source") == "sevir"
            and str(item.get("event_id", "")).upper() == normalized_id
        ),
        {"source": "sevir", "event_id": normalized_id},
    )
    result = [
        dict(item) for item in descriptions
        if not (
            item.get("source") == "sevir"
            and str(item.get("event_id", "")).upper() == normalized_id
        )
    ]
    existing["selection"] = "explicit_event_id"
    if include_observations:
        existing["representation"] = "observation_sequence"
        if observation_window_minutes is not None:
            existing["observation_window_minutes"] = [
                int(observation_window_minutes[0]),
                int(observation_window_minutes[1]),
            ]
    result.append(existing)
    return normalize_data_descs(result)
