"""Resolve uploaded and catalog assets before constructing an Agent request."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..agent.contracts import ExecutionRequest
from .execution import build_execution_request, data_descs_from_payload
from .uploads import uploaded_event_match


# Structured scientific inputs remain verbatim for the Agent to interpret.
EXPLICIT_AGENT_INPUT_FIELDS = (
    "place_name", "requested_time_utc", "aurora_initialization_time_utc",
    "latitude", "longitude", "region", "aurora_region", "forecast_leads_hours",
    "history_before_hours", "history_after_hours", "weather_model",
)


def bind_request_assets(
    payload: dict[str, Any],
    *,
    selected_subject: dict[str, Any] | None = None,
) -> ExecutionRequest:
    """Bind explicit assets, or a prior selection for an admitted continuation."""

    resolution_source = "explicit_request"
    if selected_subject and not any(payload.get(key) for key in (
        "event_id", "sevir_subject_key", "upload_id", "sequence_upload_id", "data_descs",
    )):
        if selected_subject.get("event_id"):
            payload["event_id"] = selected_subject["event_id"]
            if selected_subject.get("subject_key"):
                payload["sevir_subject_key"] = selected_subject["subject_key"]
            resolution_source = "study_case_decision"

    event_id = str(payload.get("event_id") or "").strip().upper()
    # Resolve upload identity before construction so every capability uses
    # the same asset from the first execution step.
    upload_id = str(
        payload.get("sequence_upload_id") or payload.get("upload_id") or ""
    ).strip()
    if upload_id:
        matched = uploaded_event_match(upload_id)
        event_match = matched.get("event_match", {})
        resolved_event = str(event_match.get("event_id") or "").strip().upper()
        if event_id and resolved_event and resolved_event != event_id:
            raise ValueError(
                f"request event_id {event_id} conflicts with upload event_id {resolved_event}"
            )
        if resolved_event and not event_id:
            event_id = resolved_event
            resolution_source = "bound_upload_exact_match"
        if event_match.get("subject_key"):
            explicit_subject = str(payload.get("sevir_subject_key") or "").strip().lower()
            if explicit_subject and explicit_subject != event_match["subject_key"]:
                raise ValueError("request sevir_subject_key conflicts with upload subject_key")
            payload["sevir_subject_key"] = event_match["subject_key"]

    if event_id:
        from weather_agent.tools.sevir.repository import resolve_sevir_subject_key

        subject_key = str(payload.get("sevir_subject_key") or "").strip().lower()
        if not subject_key:
            subject_key = resolve_sevir_subject_key(event_id)
        payload["event_id"] = event_id
        payload["sevir_subject_key"] = subject_key
        payload["event_resolution"] = {
            "status": "resolved",
            "event_id": event_id,
            "subject_key": subject_key,
            "source": resolution_source,
        }

    descriptions = data_descs_from_payload(payload)
    if event_id:
        matching = [
            item for item in descriptions
            if isinstance(item, dict)
            and str(item.get("source") or "").strip().lower() == "sevir"
            and str(item.get("event_id") or "").strip().upper() == event_id
        ]
        if not matching:
            matching = [{"source": "sevir", "event_id": event_id}]
            descriptions = [*descriptions, *matching]
        for item in matching:
            item.setdefault("subject_key", payload["sevir_subject_key"])
            item["selection"] = "explicit_event_id"
            item["representation"] = "event_handle"

    request = build_execution_request(
        descriptions, payload.get("selected_tools"), payload,
        generate_visual=payload.get("generate_visual", False),
        explicit_inputs={
            key: payload[key] for key in EXPLICIT_AGENT_INPUT_FIELDS
            if key in payload and payload[key] not in (None, "", [], {})
        },
    )
    payload.update(asdict(request.binding))
    payload["data_descs"] = list(request.data_descs)
    payload["selected_tools"] = list(request.selected_tools)
    return request
