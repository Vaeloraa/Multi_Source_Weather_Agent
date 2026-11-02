from __future__ import annotations

"""The complete application service for one adaptive code Agent.

This module owns request admission, context assembly and Agent orchestration.
It does not infer a weather workflow, select a response template, audit
conclusions, or execute a second Agent mode.
"""

import threading
import time
from typing import Any, Callable, Mapping

from weather_agent.agent.contracts import ExecutionRuntime, ExecutionRequest
from weather_agent.agent.unified import AgentCancelled, run_unified_agent
from weather_agent.config import (
    configured_agent_limits, model_api_timeout_seconds,
    configured_agent_request_budget_seconds,
    configured_model_concurrency,
)


Completion = Callable[
    [dict[str, str], str, list[dict[str, str]], float], dict[str, Any]
]

_AGENT_REQUEST_SLOTS = threading.BoundedSemaphore(configured_model_concurrency())


def request_budget_seconds(
    data_policy: str | None = None, *, override_seconds: float | None = None
) -> float:
    """Return the budget for a request policy.

    Cached lookups retain the 30-second interactive budget.  Live requests
    get a substantially longer budget so online retrieval/local inference can
    finish and persist its result for later cache reuse.
    """

    configured = configured_agent_request_budget_seconds(data_policy)

    if override_seconds is not None:
        return min(configured, max(1.0, float(override_seconds)))
    return configured


def _execute_unified_request_held(
    provider_name: str,
    model: str,
    question: str,
    *,
    execution_request: ExecutionRequest,
    providers: Mapping[str, dict[str, str]],
    complete: Completion,
    runtime: ExecutionRuntime,
    summarize: Completion | None = None,
) -> dict[str, Any]:
    """Run the only Agent path and return its unmodified analytical result."""

    provider = providers[provider_name]
    deadline = runtime.deadline_monotonic
    if deadline is None:
        raise ValueError("ExecutionRuntime.deadline_monotonic is required")
    total_seconds = float(
        runtime.budget_seconds or max(0.0, deadline - time.monotonic())
    )
    timings: list[dict[str, Any]] = []
    request_trace = runtime.trace
    completed_tool_calls = 0

    def timed_complete(
        current_provider: dict[str, str],
        current_model: str,
        messages: list[dict[str, str]],
        *, summary: bool = False,
    ) -> dict[str, Any]:
        if runtime.is_cancelled():
            raise AgentCancelled("client disconnected before model call")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The analysis time limit has been reached")
        if summary and remaining <= 20.0:
            raise TimeoutError("time reserved for the final answer")
        started = time.perf_counter()
        error: Exception | None = None
        try:
            completion = summarize if summary else complete
            response = completion(
                current_provider,
                current_model,
                messages,
                min(remaining - 20.0 if summary else remaining, model_api_timeout_seconds()),
            )
            if runtime.is_cancelled():
                raise AgentCancelled("client disconnected during model call")
            if time.monotonic() >= deadline:
                raise TimeoutError("The model call exceeded the remaining analysis time")
            return response
        except Exception as exc:
            error = exc
            raise
        finally:
            timings.append({
                "kind": "context_summary" if summary else "agent_step",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "error" if error else "ok",
                "error_type": type(error).__name__ if error else None,
                "remaining_ms_after": round(
                    max(0.0, deadline - time.monotonic()) * 1000, 2
                ),
            })

    def summarize_context(messages: list[dict[str, str]]) -> str:
        response = timed_complete(provider, model, messages, summary=True)
        if response["choices"][0].get("finish_reason") == "length":
            raise ValueError("context summary was truncated")
        return response["choices"][0]["message"]["content"]

    # Live acquisition/inference is allowed a larger reasoning/execution
    # envelope.  Cached follow-ups keep the tighter interactive defaults.
    limits = configured_agent_limits()
    agent_max_steps = limits["max_steps"]

    def publish_agent_progress(event: dict[str, Any]) -> None:
        """Translate internal Agent events into a bounded public request timeline."""

        nonlocal completed_tool_calls
        if request_trace is None:
            return
        attempt = max(0, int(event.get("attempt") or event.get("step", {}).get("attempt") or 0))
        progress = min(0.93, 0.17 + (0.72 * attempt / max(1, agent_max_steps)))
        if event.get("event") == "step_completed":
            step = event["step"]
            completed_tool_calls += len(step.get("calls", []))
            request_trace.append_live_step(step, progress=progress)
            request_trace.add(
                current_stage="agent_step_completed",
                model_calls=max(int(step.get("attempt") or 0), 0),
                tool_calls=completed_tool_calls,
            )
            return
        action = str(event.get("action") or "prepare")
        stage = {
            "model": "agent_model",
            "execute": "agent_execution",
            "prepare": "unified_agent",
            "context": "agent_context",
        }.get(action, "unified_agent")
        request_trace.update_activity(
            stage=stage,
            summary={
                "model": "Using the available results to choose the next step.",
                "execute": "Running analysis code and reading tool results.",
                "prepare": "Preparing the agent.",
                "context": "Preparing conversation context and previous results.",
            }.get(action, "Analysis in progress."),
            progress=progress,
            attempt=attempt,
            action=action,
            status=str(event.get("status") or "running"),
            model_calls=int(event.get("model_calls") or attempt),
            execution_attempts=int(event.get("execution_attempts") or 0),
        )

    result = run_unified_agent(
        provider,
        model,
        question,
        timed_complete,
        execution_request=execution_request,
        progress_callback=publish_agent_progress,
        # Model turns, code executions and consecutive failures are separate
        # safety budgets.  A multi-source comparison must retain room for a
        # final synthesis after its last useful observation.
        max_steps=agent_max_steps,
        max_executions=limits["max_executions"],
        max_error_retries=limits["max_error_retries"],
        runtime=runtime,
        summarize=summarize_context if summarize is not None else None,
    )
    result["session_id"] = runtime.session_id
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
    *,
    execution_request: ExecutionRequest,
    providers: Mapping[str, dict[str, str]],
    complete: Completion,
    runtime: ExecutionRuntime,
    summarize: Completion | None = None,
) -> dict[str, Any]:
    """Admit one Agent request before starting its own execution budget.

    Queue time is part of the same end-to-end request budget. Once admitted,
    all Reflective model calls for that request share the unchanged deadline.
    """

    queue_started = time.perf_counter()
    acquired = False
    request_trace = runtime.trace
    try:
        queue_deadline = runtime.deadline_monotonic
        if queue_deadline is None:
            raise ValueError("ExecutionRuntime.deadline_monotonic is required")
        if request_trace is not None:
            request_trace.update_activity(
                stage="agent_queue",
                summary="Waiting for an available execution slot.",
                progress=0.12,
                action="queue",
            )
        while not acquired:
            if runtime.is_cancelled():
                raise AgentCancelled("client disconnected while waiting for Agent slot")
            remaining = queue_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The execution queue exceeded the remaining request time")
            acquired = _AGENT_REQUEST_SLOTS.acquire(timeout=min(0.2, remaining))
        admitted_at = time.perf_counter()
        if request_trace is not None:
            queue_wait_ms = round((admitted_at - queue_started) * 1000, 2)
            request_trace.update_activity(
                stage="unified_agent",
                summary=(
                    f"Execution slot acquired after {queue_wait_ms / 1000:.1f} s. Starting the agent."
                    if queue_wait_ms >= 1000
                    else "Execution slot acquired. Starting the agent."
                ),
                progress=0.16,
                action="prepare",
                queue_wait_ms=queue_wait_ms,
            )
        requested_policy = execution_request.binding.data_policy
        result = _execute_unified_request_held(
            provider_name,
            model,
            question,
            execution_request=execution_request,
            providers=providers,
            complete=complete,
            runtime=runtime, summarize=summarize,
        )
        result["cache_resolution"] = {
            "strategy": (
                "capability_exact_cache_then_live"
                if requested_policy == "live"
                else "capability_exact_cache_only"
            ),
            "requested_policy": requested_policy,
            "live_allowed": requested_policy == "live",
        }
        result["request_admission"] = {
            "queue_wait_ms": round((admitted_at - queue_started) * 1000, 2),
            "slot_count": 1,
        }
        return result
    finally:
        if acquired:
            _AGENT_REQUEST_SLOTS.release()
