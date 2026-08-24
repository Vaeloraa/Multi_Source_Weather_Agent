from __future__ import annotations

"""The complete application service for one adaptive code Agent.

This module deliberately owns only request asset binding and orchestration.
It does not infer a weather workflow, select a response template, audit
conclusions, or execute a second Agent mode.
"""

import os
import re
import time
from typing import Any, Callable, Mapping

from sevir_world.agent.contracts import data_descs_for_event, data_descs_from_context
from sevir_world.agent.session import context_for_prompt, load_session, normalize_session_id
from sevir_world.agent.unified import AgentCancelled, run_unified_agent


AGENT_CAPABILITIES = (
    "ds",
    "sevir",
    "geolocator",
    "cascast",
    "aurora",
)

Completion = Callable[
    [dict[str, str], str, list[dict[str, str]], float], dict[str, Any]
]

MAX_AGENT_REQUEST_SECONDS = 30.0


def request_budget_seconds() -> float:
    """Return the configured request budget, capped at the 30-second contract."""

    try:
        configured = float(
            os.getenv("SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS", "30")
        )
    except ValueError:
        configured = MAX_AGENT_REQUEST_SECONDS
    return min(MAX_AGENT_REQUEST_SECONDS, max(1.0, configured))


def bind_request_assets(
    question: str, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve explicit asset identities without deciding the analysis."""

    event_id = str(payload.get("event_id", "")).strip().upper()
    if not event_id:
        match = re.search(r"(?<![A-Za-z0-9])([A-Za-z]\d{6,})(?![A-Za-z0-9])", question)
        event_id = match.group(1).upper() if match else ""
    if event_id:
        payload["event_id"] = event_id
        payload["event_resolution"] = {
            "status": "resolved",
            "event_id": event_id,
            "source": "explicit_request",
        }

    descriptions = data_descs_from_context(payload)
    if event_id:
        descriptions = data_descs_for_event(
            event_id,
            descriptions,
            include_observations=False,
        )
        for item in descriptions:
            if item.get("source") == "sevir" and item.get("event_id") == event_id:
                item["representation"] = "event_handle"

    selected = payload.get("selected_tools")
    if isinstance(selected, list) and selected:
        tools = ["ds"]
        for item in selected:
            requested = str(item).strip()
            if requested not in AGENT_CAPABILITIES:
                raise ValueError(f"unknown capability: {requested}")
            if requested not in tools:
                tools.append(requested)
    else:
        tools = list(AGENT_CAPABILITIES)

    payload["data_descs"] = descriptions
    payload["selected_tools"] = tools
    # Reject stale client/server state from the retired architecture at the
    # one ingress boundary instead of letting it influence the Agent prompt.
    for key in (
        "task_spec",
        "execution_plan",
        "capability_request",
        "required_evidence",
        "_compiled_request",
        "mode",
        "resolved_mode",
    ):
        payload.pop(key, None)
    return descriptions, tools


def execute_unified_request(
    provider_name: str,
    model: str,
    question: str,
    payload: dict[str, Any],
    *,
    providers: Mapping[str, dict[str, str]],
    complete: Completion,
) -> dict[str, Any]:
    """Run the only Agent path and return its unmodified analytical result."""

    if provider_name not in providers:
        raise ValueError("不支持的大模型提供方。")
    provider = providers[provider_name]
    session_id = normalize_session_id(payload.get("session_id"))
    descriptions = list(payload.get("data_descs") or [])
    tools = list(payload.get("selected_tools") or AGENT_CAPABILITIES)
    total_seconds = request_budget_seconds()
    now = time.monotonic()
    supplied_deadline = payload.get("_deadline_monotonic")
    try:
        deadline = min(float(supplied_deadline), now + total_seconds)
    except (TypeError, ValueError):
        deadline = now + total_seconds
    cancelled = payload.get("_cancelled")
    is_cancelled = cancelled if callable(cancelled) else (lambda: False)
    timings: list[dict[str, Any]] = []

    def timed_complete(
        current_provider: dict[str, str],
        current_model: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        if is_cancelled():
            raise AgentCancelled("client disconnected before model call")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("统一Agent本轮总时间已达到上限")
        started = time.perf_counter()
        error: Exception | None = None
        try:
            response = complete(
                current_provider,
                current_model,
                messages,
                max(0.1, remaining),
            )
            if is_cancelled():
                raise AgentCancelled("client disconnected during model call")
            if time.monotonic() >= deadline:
                raise TimeoutError("统一Agent模型调用超过本轮总时间上限")
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            timings.append({
                "kind": "agent_step",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "error" if error else "ok",
                "error_type": type(error).__name__ if error else None,
                "remaining_ms_after": round(
                    max(0.0, deadline - time.monotonic()) * 1000, 2
                ),
            })

    context = {
        "_deadline_monotonic": deadline,
        "_cancelled": is_cancelled,
        "event_id": str(payload.get("event_id", "")),
        "data_source": str(payload.get("data_source", "cached")),
        "data_policy": str(payload.get("data_policy", "cached")),
        "weather_model": str(payload.get("weather_model", "cascast")),
        "analysis_time_utc": str(payload.get("analysis_time_utc", "")),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "region": payload.get("region"),
        "forecast_leads_hours": payload.get("forecast_leads_hours"),
        "history_before_hours": payload.get("history_before_hours"),
        "history_after_hours": payload.get("history_after_hours"),
        "live_preflight": payload.get("live_preflight"),
        "upload_id": str(payload.get("upload_id", "")),
        "sequence_upload_id": str(payload.get("sequence_upload_id", "")),
    }
    result = run_unified_agent(
        provider,
        model,
        question,
        context,
        timed_complete,
        data_descs=descriptions,
        selected_tools=tools,
        conversation_context=context_for_prompt(load_session(session_id)),
        max_steps=int(os.getenv("SEVIR_UNIFIED_MAX_STEPS", "8")),
        max_error_retries=int(os.getenv("SEVIR_UNIFIED_MAX_ERROR_RETRIES", "3")),
    )
    result["session_id"] = session_id
    result["llm_timings"] = timings
    result["request_budget"] = {
        "limit_seconds": total_seconds,
        "remaining_ms": round(max(0.0, deadline - time.monotonic()) * 1000, 2),
        "deadline_reached": time.monotonic() >= deadline,
    }
    return result
