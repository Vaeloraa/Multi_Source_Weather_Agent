from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..capabilities.base import RequestBinding


CANONICAL_ARGUMENTS = (
    "ds",
    "sevir",
    "geolocator",
    "cascast",
    "aurora",
)
DEFAULT_SELECTED_TOOLS = CANONICAL_ARGUMENTS
DATA_CONTRACT_VERSION = "sevir-data-v3"
# Bump this whenever the injected namespace or public capability return
# contracts change. The persistent Worker uses it as an explicit generation
# token and must not be reused across generations.
WORKER_CONTRACT_VERSION = "sevir-worker-v1"
_DATA_SOURCES = {"upload", "sevir"}


@dataclass(frozen=True)
class RequestContext:
    """Canonical immutable request facts resolved before Agent execution."""

    data_descs: tuple[dict[str, Any], ...]
    selected_tools: tuple[str, ...]
    binding: RequestBinding
    visual_artifact_required: bool = False


@dataclass(frozen=True)
class ExecutionRequest:
    """One canonical request context plus cross-request evidence state."""

    context: RequestContext
    evidence_state: dict[str, Any] = field(default_factory=dict)

    @property
    def data_descs(self) -> tuple[dict[str, Any], ...]:
        return self.context.data_descs

    @property
    def selected_tools(self) -> tuple[str, ...]:
        return self.context.selected_tools

    @property
    def binding(self) -> RequestBinding:
        return self.context.binding

    @property
    def visual_artifact_required(self) -> bool:
        return self.context.visual_artifact_required

    @classmethod
    def from_payload(
        cls,
        data_descs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        selected_tools: list[str] | tuple[str, ...] | None,
        binding: dict[str, Any] | RequestBinding | None = None,
        evidence_state: dict[str, Any] | None = None,
        visual_artifact_required: bool = False,
    ) -> "ExecutionRequest":
        raw = normalize_data_descs(data_descs)
        priority = {"upload": 0, "sevir": 1}
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
        if isinstance(binding, RequestBinding):
            bound = binding
        else:
            inferred = dict(binding or {})
            for item in descriptions:
                if item.get("source") == "upload" and not inferred.get("upload_id"):
                    inferred["upload_id"] = item.get("upload_id")
                if item.get("source") == "sevir" and not inferred.get("event_id"):
                    inferred["event_id"] = item.get("event_id")
                if item.get("source") == "sevir" and not inferred.get("sevir_subject_key"):
                    inferred["sevir_subject_key"] = item.get("subject_key")
            bound = RequestBinding.from_mapping(inferred)
        bound = _reconcile_asset_identities(descriptions, bound)
        return cls(
            RequestContext(
                descriptions,
                ordered,
                bound,
                bool(visual_artifact_required),
            ),
            dict(evidence_state or {}),
        )


def _reconcile_asset_identities(
    descriptions: tuple[dict[str, Any], ...],
    binding: RequestBinding,
) -> RequestBinding:
    """Make loaded records and request-bound capabilities share one identity.

    A request may contain both an upload and its matched SEVIR event, but it
    must never bind a capability to a different identity than the records of
    the same source injected as ``ds``.
    """

    updates: dict[str, str] = {}
    for source, field_name, description_field in (
        ("upload", "upload_id", "upload_id"),
        ("sevir", "event_id", "event_id"),
        ("sevir", "sevir_subject_key", "subject_key"),
    ):
        identities = tuple(dict.fromkeys(
            str(item.get(description_field) or "").strip()
            for item in descriptions
            if item.get("source") == source and item.get(description_field)
        ))
        bound_value = str(getattr(binding, field_name) or "").strip()
        if field_name == "event_id":
            bound_value = bound_value.upper()
        if bound_value and identities and bound_value not in identities:
            raise ValueError(
                f"request identity conflict: binding.{field_name}={bound_value!r} "
                f"is not present in data_descs {field_name} values {identities!r}"
            )
        if not bound_value and identities:
            updates[field_name] = identities[0]
    return replace(binding, **updates) if updates else binding


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
                "upload, sevir"
            )
        item["source"] = source
        key_name = "upload_id" if source == "upload" else "event_id"
        identity_value = str(item.get(key_name, "")).strip()
        if not identity_value:
            raise ValueError(f"data_descs[{index}] requires {key_name}")
        if source != "upload":
            identity_value = identity_value.upper()
        item[key_name] = identity_value
        if source == "sevir" and item.get("subject_key"):
            item["subject_key"] = str(item["subject_key"]).strip().lower()
        identity = (source, str(item.get("subject_key") or identity_value))
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
    descriptions: list[dict[str, Any]] = []
    upload_id = bound.get("sequence_upload_id") or bound.get("upload_id")
    if upload_id:
        descriptions.append({"source": "upload", "upload_id": str(upload_id)})
    if bound.get("event_id"):
        event_description = {
            "source": "sevir",
            "event_id": str(bound["event_id"]),
        }
        if bound.get("sevir_subject_key"):
            event_description["subject_key"] = str(bound["sevir_subject_key"])
        descriptions.append(event_description)
    return normalize_data_descs(descriptions)


def request_binding_from_context(context: dict[str, Any] | None) -> RequestBinding:
    """Build the only request identity passed to capability constructors."""

    value = dict(context or {})
    preflight = value.get("live_preflight")
    if isinstance(preflight, dict):
        for key in (
            "requested_time_utc", "aurora_initialization_time_utc",
            "latitude", "longitude", "region", "aurora_region",
            "forecast_leads_hours", "history_before_hours", "history_after_hours",
        ):
            if value.get(key) is None and preflight.get(key) is not None:
                value[key] = preflight[key]
    return RequestBinding.from_mapping(value)


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
    if not existing.get("subject_key"):
        try:
            from ..tools.sevir.Catalog import resolve_sevir_subject_key

            existing["subject_key"] = resolve_sevir_subject_key(normalized_id)
        except (ValueError, RuntimeError):
            # The request ingress owns ambiguity errors.  This helper may also
            # be used while remote catalog metadata is temporarily absent.
            pass
    if include_observations:
        existing["representation"] = "observation_sequence"
        if observation_window_minutes is not None:
            existing["observation_window_minutes"] = [
                int(observation_window_minutes[0]),
                int(observation_window_minutes[1]),
            ]
    result.append(existing)
    return normalize_data_descs(result)
