from __future__ import annotations

"""Minimal response boundary for the single unified Agent.

The unified architecture has one owner of analytical decisions—the Agent. This module
only adds HTTP metadata, persists the conversation, and closes the request
trace.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable

from sevir_world.agent.session import (
    append_turn,
    normalize_session_id,
    normalize_turn_id,
)
from sevir_world.agent.study import build_turn_evidence, evidence_index


EvidenceSummaryBuilder = Callable[[dict[str, Any]], dict[str, Any]]


def _public_step(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    calls = []
    for call in item.get("calls", []) if isinstance(item.get("calls"), list) else []:
        if isinstance(call, dict):
            calls.append({
                "api": call.get("api"),
                "status": call.get("status"),
                "cache_hit": bool(call.get("cache_hit", False)),
                "cost_units": call.get("cost_units", 0.0),
                "arguments": call.get("arguments", {}),
                "result_summary": call.get("result_summary"),
                "error": call.get("error"),
            })
    return {
        "step": item.get("attempt"),
        "action": item.get("action"),
        "status": item.get("status"),
        "summary": item.get("summary"),
        "error": item.get("error"),
        "elapsed_ms": item.get("elapsed_ms"),
        "model_elapsed_ms": item.get("model_elapsed_ms"),
        "tools": calls,
        "evidence_acquired": bool(item.get("evidence_acquired", False)),
        "postprocessing_succeeded": bool(item.get("postprocessing_succeeded", False)),
        "tool_cost_summary": item.get("tool_cost_summary"),
    }


def _persist_turn(
    result: dict[str, Any], payload: dict[str, Any], question: str
) -> None:
    session_id = result.get("session_id") or payload.get("session_id")
    if not session_id:
        return
    session_id = normalize_session_id(session_id)
    turn_id = normalize_turn_id(
        result.get("session_turn_id") or payload.get("turn_id")
    )
    evidence_record = build_turn_evidence(
        turn_id=turn_id,
        request_id=str(result.get("request_id") or payload.get("request_id") or ""),
        question=question,
        payload=payload,
        result=result,
    )
    saved = append_turn(
        session_id,
        {
            "turn_id": turn_id,
            "question": question,
            "answer": str(result.get("answer", "")),
            "selected_tools": list(result.get("selected_tools", [])),
            "termination_reason": str(result.get("termination_reason", "unknown")),
            "code_execution_count": int(result.get("code_execution_count", 0) or 0),
            "evidence_execution_count": int(result.get("evidence_execution_count", 0) or 0),
            "evidence_tool_call_count": int(result.get("evidence_tool_call_count", 0) or 0),
            "postprocessing_success_count": int(result.get("postprocessing_success_count", 0) or 0),
            "tool_cost_summary": dict(result.get("tool_cost_summary") or {}),
            "artifacts": list(result.get("artifacts", [])),
            # Execution provenance supports follow-ups; this is neither an
            # answer cache nor a precomputed analytical conclusion.
            "observations": list(result.get("observations", [])),
            "evidence_record": evidence_record,
            "study_case_id": str(payload.get("study_case_id") or session_id),
            "active_upload": {
                "upload_id": str(payload.get("upload_id", "")),
                "sequence_upload_id": str(payload.get("sequence_upload_id", "")),
                "data_source": str(payload.get("data_source", "upload")),
            },
        },
    )
    result["session_id"] = session_id
    result["session_turn_id"] = turn_id
    result["session_turn_count"] = len(saved.get("turns", []))
    result["study_case"] = evidence_index(saved.get("study_case"))


def finalize_agent_response(
    result: dict[str, Any],
    *,
    payload: dict[str, Any],
    question: str,
    request_id: str,
    mode: str,
    provider_name: str,
    model_name: str,
    route_decision: dict[str, Any],
    started: float,
    analysis_started: float,
    trace: Any,
    presentation: dict[str, Any],
    artifact_root: Path,
    evidence_summary_builder: EvidenceSummaryBuilder | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Attach transport metadata without changing the Agent's conclusion."""

    result.update({
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "request_id": request_id,
        "mode": "unified",
        "requested_mode": route_decision.get("requested_mode", "auto"),
        "route_decision": route_decision,
        "provider": provider_name,
        "model": model_name,
        "event_resolution": payload.get("event_resolution"),
        "presentation": presentation,
        "artifact_count": len(result.get("artifacts", [])),
    })
    public_trace = [
        step
        for item in result.get("agent_trace", [])
        if (step := _public_step(item)) is not None
    ]
    # Expose one human-readable execution timeline. The internal trace contains
    # generated code and nested evidence JSON; observations keep that material
    # server-side for reflection and StudyEvidence, while the browser receives
    # only actions, timings and concise tool descriptions.
    result["agent_trace"] = public_trace
    result.pop("progress_steps", None)

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
            if isinstance(call, dict) and call.get("cache_hit") is True
        ),
        termination_reason=result.get("termination_reason"),
        route_mode=result.get("route_mode"),
    )
    is_cancelled = cancelled if callable(cancelled) else (lambda: False)
    within_budget = (
        deadline_monotonic is None or time.monotonic() < deadline_monotonic
    )
    if not is_cancelled() and within_budget:
        _persist_turn(result, payload, question)
    else:
        result["finalization_deferred"] = (
            "client_cancelled" if is_cancelled() else "request_deadline_reached"
        )
    result["request_trace"] = trace.finish(str(result.get("status", "ok")))

    if bool(payload.get("trace", False)):
        result["trace"] = {
            "request_id": request_id,
            "question": question,
            "event_id": str(payload.get("event_id", "")),
            "selected_tools": list(result.get("selected_tools", [])),
            "steps": public_trace,
            "termination_reason": result.get("termination_reason"),
            "answer": str(result.get("answer", "")),
            "finalization_deferred": result.get("finalization_deferred"),
            "request_trace": result.get("request_trace"),
        }
        trace_path = artifact_root / f"{request_id}.trace.json"
        try:
            trace_path.write_text(
                json.dumps(result["trace"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result["trace_file"] = str(trace_path)
        except OSError:
            result["trace_file"] = None
    return result
