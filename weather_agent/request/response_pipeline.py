from __future__ import annotations

"""Minimal response boundary for the single unified Agent.

The unified architecture has one owner of analytical decisions—the Agent. This module
only adds HTTP metadata, persists the conversation, and closes the request
trace.
"""

import re
import time
from typing import Any

from weather_agent.errors import classify_failure, failure_class_for_termination, failure_guidance
from weather_agent.agent.session import build_turn_commit
from weather_agent.capabilities.registry import operation_display


def _attach_registered_artifacts(answer: str, artifacts: list[dict[str, Any]]) -> str:
    """Attach current-request artifacts exactly once, independent of LLM prose."""
    text = str(answer or "").strip()
    registered = {item["url"] for item in artifacts}
    # Link repair belongs at the presentation boundary and needs no model turn.
    text = re.sub(
        r"(!?)\[([^\]]*)\]\((/api/artifacts/[^\s)]+)\)",
        lambda match: match.group(0) if match.group(3) in registered
        else "" if match.group(1) else match.group(2),
        text,
    )
    text = re.sub(
        r"/api/artifacts/[^\s)\]>'\"]+",
        lambda match: match.group(0) if match.group(0) in registered else "",
        text,
    )
    existing = set(re.findall(r"/api/artifacts/[0-9a-f]{32}\.png", text, re.IGNORECASE))
    links: list[str] = []
    for artifact in artifacts:
        url = artifact["url"]
        if url in existing:
            continue
        title = str(artifact.get("title") or artifact.get("artifact_id") or "Analysis chart")
        links.append(f"![{title}]({url})")
        existing.add(url)
    if links:
        text = (text + "\n\n" if text else "") + "\n".join(links)
    return text.strip()


def agent_step_summary(step: dict[str, Any]) -> str:
    """Render technical Agent events for live and completed timelines."""
    if step.get("summary"):
        return str(step["summary"])
    if step.get("status") == "cancelled":
        return "Request cancelled."
    action = step.get("action")
    if action == "execute":
        return {"ok": "Execution completed. Expand to inspect the returned results.", "partial": "Code completed, but some tools returned no results."}.get(
            step.get("status"), "Execution failed. Error details were recorded.",
        )
    return {
        "model": "Model request incomplete.",
        "select_tools": "Tools selected and interface instructions loaded.",
        "read_context": "Previous analysis records loaded.",
        "context_budget": "The question and required instructions exceed the context budget.",
        "format_repair": "Invalid action format. A corrected action was requested.",
        "citation_repair": "Adding citations and checking source passages.",
        "knowledge_reuse": "Reusing retrieved documents and reading the remaining passages.",
        "finish": "Answer completed from the available results." if step.get("status") == "ok" else "Available results summarized. Analysis remains incomplete.",
        "stop": "Analysis stopped. Completed results were retained.",
    }.get(action, "Analyzing.")


def _public_step(item: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for call in item.get("calls", []):
        display = operation_display(call["api"])
        calls.append({
            "api": call["api"],
            **{key: call[key] for key in ("call_id", "parent_call_id", "phase", "backend_executed", "work") if key in call},
            "display_name": call.get("display_name", display["display_name"]),
            "stage": call.get("stage", display["stage"]),
            "model_id": call.get("model_id"),
            "product": call.get("product"),
            "status": call.get("status"),
            "cache_hit": bool(call.get("cache_hit", False)),
            "cost_units": call.get("cost_units", 0.0),
            "arguments": call.get("arguments", {}),
            "result_summary": call.get("result_summary"),
            "error": call.get("error"),
            **({"retrieval": call["retrieval"]} if "retrieval" in call else {}),
        })
    return {
        "attempt": item.get("attempt"),
        "step": item.get("step", item.get("attempt")),
        "action": item.get("action"),
        **({"selected_tools": item["selected_tools"]} if "selected_tools" in item else {}),
        "status": item.get("status"),
        "summary": agent_step_summary(item),
        **({"execution_feedback": item["execution_feedback"]} if "execution_feedback" in item else {}),
        "error": item.get("error"),
        **({"rejection_code": item["rejection_code"]} if "rejection_code" in item else {}),
        **({"rejection_details": item["rejection_details"]} if "rejection_details" in item else {}),
        **({"execution_started": item["execution_started"]} if "execution_started" in item else {}),
        "elapsed_ms": item.get("elapsed_ms"),
        "execution_elapsed_ms": item.get("execution_elapsed_ms"),
        "model_elapsed_ms": item.get("model_elapsed_ms"),
        "total_elapsed_ms": item.get("total_elapsed_ms", item.get("elapsed_ms")),
        "calls": calls,
        "tools": calls,
        "evidence_acquired": bool(item.get("evidence_acquired", False)),
        "postprocessing_succeeded": bool(item.get("postprocessing_succeeded", False)),
        "tool_cost_summary": item.get("tool_cost_summary"),
    }


def result_outcome(status: Any) -> dict[str, Any]:
    """One public projection of analytical status, independent of transport state."""
    normalized = str(status or "unknown").strip().lower()
    labels = {
        "ok": "Completed", "success": "Completed", "completed": "Completed",
        "partial": "Partially completed", "cancelled": "Cancelled", "canceled": "Cancelled",
        "interrupted": "Interrupted", "error": "Incomplete", "failed": "Incomplete",
        "failure": "Incomplete", "unavailable": "Data unavailable", "blocked": "Incomplete",
        "clarification_needed": "More information needed", "unknown": "Result pending",
    }
    return {"status": normalized, "label": labels.get(normalized, "Result pending"),
            "incomplete": normalized not in {"ok", "success", "completed"}}


def finalize_agent_response(
    result: dict[str, Any],
    *,
    payload: dict[str, Any],
    question: str,
    request_id: str,
    provider_name: str,
    model_name: str,
    started: float,
    analysis_started: float,
    trace: Any,
    presentation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Format one response and prepare its conversation turn for atomic saving."""

    result["answer"] = _attach_registered_artifacts(
        str(result.get("answer", "")),
        result.get("artifacts", []),
    )
    result.update({
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "request_id": request_id,
        "mode": "unified",
        "provider": provider_name,
        "model": model_name,
        "event_resolution": payload.get("event_resolution"),
        "presentation": {**presentation, "outcome": result_outcome(result.get("status"))},
        "artifact_count": len(result.get("artifacts", [])),
    })
    public_trace = [_public_step(item) for item in result.get("agent_trace", [])]
    # Expose one human-readable execution timeline. The internal trace contains
    # generated code and nested evidence JSON; observations keep that material
    # server-side for reflection and StudyEvidence, while the browser receives
    # actions, timings, concise tool descriptions and bounded result previews.
    result["agent_trace"] = public_trace

    trace.record_stage(
        "unified_agent",
        (time.perf_counter() - analysis_started) * 1000,
        status=str(result.get("status", "ok")),
        termination_reason=result.get("termination_reason"),
        model_calls=int(result.get("model_calls", 0) or 0),
        code_executions=int(result.get("code_execution_count", 0) or 0),
    )
    trace.add(
        model_calls=int(result.get("model_calls", 0) or 0),
        tool_calls=len(result.get("tool_runs", [])),
        cache_hits=sum(
            1 for call in result.get("tool_runs", [])
            if call.get("cache_hit") is True
        ),
        termination_reason=result.get("termination_reason"),
    )
    failure_class = str(result.get("failure_class") or "").strip()
    if not failure_class and str(result.get("status", "ok")).lower() in {
        "error", "failed", "partial", "unavailable",
    }:
        raw_error = result.get("error")
        # A provider's structured error is more specific than the Agent's
        # generic termination reason (for example quota vs transport failure).
        if isinstance(raw_error, dict):
            failure_class = classify_failure(raw_error)
        if not failure_class or failure_class == "unknown_error":
            failure_class = failure_class_for_termination(result.get("termination_reason")) or classify_failure(
                raw_error or result.get("termination_reason") or "unknown_error",
                str(result.get("termination_reason") or "unified_agent"),
            )
    if failure_class:
        failure_message, next_action = failure_guidance(failure_class)
        result["failure_class"] = failure_class
        failure_step = public_trace[-1] if public_trace else {}
        failure_phase = result.get("failure_phase") or failure_step.get("action") or "unified_agent"
        result["failure_phase"] = failure_phase
        result.setdefault("failure_message", failure_message)
        result.setdefault("next_action", next_action)
        trace.add(
            failure_class=failure_class,
            failure_phase=failure_phase,
            failure_message=result["failure_message"],
            next_action=result["next_action"],
        )
    turn_commit = build_turn_commit(result, payload, question)
    # Source archives and derived memory remain server-side in the turn record.
    result.pop("context_summary", None)
    result.pop("context_archive", None)
    result["request_trace"] = trace.finish(str(result.get("status", "ok")))
    return result, turn_commit
