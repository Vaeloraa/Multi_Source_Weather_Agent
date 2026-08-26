from __future__ import annotations

"""The complete application service for one adaptive code Agent.

This module deliberately owns only request asset binding and orchestration.
It does not infer a weather workflow, select a response template, audit
conclusions, or execute a second Agent mode.
"""

import os
import re
import threading
import time
from typing import Any, Callable, Mapping

from sevir_world.agent.contracts import (
    ExecutionRequest,
    data_descs_for_event,
    data_descs_from_context,
    request_binding_from_context,
)
from sevir_world.agent.session import context_for_prompt, load_session, normalize_session_id
from sevir_world.agent.unified import AgentCancelled, run_unified_agent
from sevir_world.agent.study import runtime_state
from sevir_world.data_access.uploads import uploaded_event_match


AGENT_CAPABILITIES = (
    "ds",
    "sevir",
    "geolocator",
    "cascast",
    "aurora",
)


def _default_capabilities(payload: dict[str, Any]) -> list[str]:
    """Expose all four lazy public capabilities without choosing a workflow."""

    del payload
    return list(AGENT_CAPABILITIES)

Completion = Callable[
    [dict[str, str], str, list[dict[str, str]], float], dict[str, Any]
]

# The HTTP client, server admission deadline and Agent must share one hard
# ceiling.  A larger server-only environment value previously let the server
# continue for 150 seconds after the acceptance client had already timed out at
# 90 seconds, losing the response and its diagnostics.
AGENT_REQUEST_BUDGET_SECONDS = 90.0


def _agent_request_slots() -> threading.BoundedSemaphore:
    """Create the bounded in-flight Agent request admission gate."""

    try:
        configured = int(os.getenv("SEVIR_MODEL_CONCURRENCY", "1"))
    except ValueError:
        configured = 1
    return threading.BoundedSemaphore(max(1, min(4, configured)))


_AGENT_REQUEST_SLOTS = _agent_request_slots()


def request_budget_seconds() -> float:
    """Return the request budget, never exceeding the shared 90-second contract."""

    try:
        configured = float(
            os.getenv("SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS", "90")
        )
    except ValueError:
        configured = AGENT_REQUEST_BUDGET_SECONDS
    return min(AGENT_REQUEST_BUDGET_SECONDS, max(1.0, configured))


def bind_request_assets(
    question: str, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve explicit asset identities without deciding the analysis."""

    event_id = str(payload.get("event_id", "")).strip().upper()
    # Inside a structured Study Case, an event identifier mentioned in prose
    # may be a candidate being evaluated rather than a binding instruction.
    # Only promoted case decisions or explicit payload fields may bind it
    # across those requests.
    if not event_id and not str(payload.get("study_case_id") or "").strip():
        match = re.search(r"(?<![A-Za-z0-9])([A-Za-z]\d{6,})(?![A-Za-z0-9])", question)
        event_id = match.group(1).upper() if match else ""
    # Resolve a request-bound upload before capability construction.  The
    # result is deterministic (filename/metadata or exact cached pixels), so
    # this is asset binding rather than an analysis decision.  It makes the
    # event identity available to every capability from the first execute
    # step; a later ``match_upload`` call remains a normal evidence lookup.
    upload_id = str(
        payload.get("sequence_upload_id") or payload.get("upload_id") or ""
    ).strip()
    if upload_id:
        matched = uploaded_event_match(upload_id)
        event_match = matched.get("event_match") if isinstance(matched, dict) else None
        resolved_event = (
            str(event_match.get("event_id") or "").strip().upper()
            if isinstance(event_match, dict)
            else ""
        )
        if event_id and resolved_event and resolved_event != event_id:
            raise ValueError(
                f"request event_id {event_id} conflicts with upload event_id {resolved_event}"
            )
        if resolved_event and not event_id:
            event_id = resolved_event
            payload["event_resolution"] = {"source": "bound_upload_exact_match"}
        if isinstance(event_match, dict) and event_match.get("subject_key"):
            payload["sevir_subject_key"] = str(event_match["subject_key"]).lower()

    if event_id:
        from sevir_world.tools.sevir.Catalog import resolve_sevir_subject_key

        subject_key = str(payload.get("sevir_subject_key") or "").strip().lower()
        if not subject_key:
            subject_key = resolve_sevir_subject_key(event_id)
        payload["event_id"] = event_id
        payload["sevir_subject_key"] = subject_key
        existing_resolution = (
            payload.get("event_resolution")
            if isinstance(payload.get("event_resolution"), dict)
            else {}
        )
        payload["event_resolution"] = {
            "status": "resolved",
            "event_id": event_id,
            "subject_key": subject_key,
            "source": str(existing_resolution.get("source") or "explicit_request"),
        }

        # Event metadata and cache manifests are evidence, not request facts.
        # They are intentionally left for capability methods to return.

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
        tools = _default_capabilities(payload)

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


def _execute_unified_request_held(
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
    session_state = load_session(session_id)
    study_context_active = bool(payload.get("_study_context_active", True))
    descriptions = list(payload.get("data_descs") or [])
    tools = list(payload.get("selected_tools") or AGENT_CAPABILITIES)
    execution_request = ExecutionRequest.from_payload(
        descriptions,
        tools,
        request_binding_from_context(payload),
        runtime_state(session_state.get("study_case")) if study_context_active else {},
        visual_artifact_required=bool(
            payload.get("visual_artifact_required", False)
        ),
    )
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
    }
    result = run_unified_agent(
        provider,
        model,
        question,
        context,
        timed_complete,
        execution_request=execution_request,
        conversation_context=(
            context_for_prompt(session_state) if study_context_active else ""
        ),
        # Model turns, code executions and consecutive failures are separate
        # safety budgets.  A multi-source comparison must retain room for a
        # final synthesis after its last useful observation.
        max_steps=int(os.getenv("SEVIR_UNIFIED_MAX_STEPS", "8")),
        max_executions=int(os.getenv("SEVIR_UNIFIED_MAX_EXECUTIONS", "5")),
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


def execute_unified_request(
    provider_name: str,
    model: str,
    question: str,
    payload: dict[str, Any],
    *,
    providers: Mapping[str, dict[str, str]],
    complete: Completion,
) -> dict[str, Any]:
    """Admit one Agent request before starting its own execution budget.

    Parallel HTTP callers may wait here, but that queue time is not charged to
    the admitted Agent's configured analysis budget.  Once admitted, all
    Reflective model calls for that request share one slot and one deadline.
    """

    cancelled = payload.get("_cancelled")
    is_cancelled = cancelled if callable(cancelled) else (lambda: False)
    queue_started = time.perf_counter()
    acquired = False
    try:
        while not acquired:
            if is_cancelled():
                raise AgentCancelled("client disconnected while waiting for Agent slot")
            acquired = _AGENT_REQUEST_SLOTS.acquire(timeout=0.2)
        admitted_at = time.perf_counter()
        # Admission starts the Agent budget, but preserve an earlier caller
        # deadline (tests, cancellation bridges, and nested service calls may
        # deliberately provide one).  Never extend an existing deadline.
        admitted_deadline = time.monotonic() + request_budget_seconds()
        supplied_deadline = payload.get("_deadline_monotonic")
        try:
            payload["_deadline_monotonic"] = min(float(supplied_deadline), admitted_deadline)
        except (TypeError, ValueError):
            payload["_deadline_monotonic"] = admitted_deadline
        result = _execute_unified_request_held(
            provider_name,
            model,
            question,
            payload,
            providers=providers,
            complete=complete,
        )
        result["request_admission"] = {
            "queue_wait_ms": round((admitted_at - queue_started) * 1000, 2),
            "slot_count": 1,
        }
        return result
    finally:
        if acquired:
            _AGENT_REQUEST_SLOTS.release()
