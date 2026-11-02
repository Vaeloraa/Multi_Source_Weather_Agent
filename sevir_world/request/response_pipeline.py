from __future__ import annotations

"""Minimal response boundary for the single unified Agent.

The unified architecture has one owner of analytical decisions—the Agent. This module
only adds HTTP metadata, persists the conversation, and closes the request
trace.
"""

import re
import time
from typing import Any

from sevir_world.errors import classify_failure, failure_class_for_termination, failure_guidance
from sevir_world.agent.session import build_turn_commit
from sevir_world.capabilities.registry import operation_display


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
        title = str(artifact.get("title") or artifact.get("artifact_id") or "分析图")
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
        return "请求已取消。"
    action = step.get("action")
    if action == "execute":
        return {"ok": "程序运行成功；本步返回值可展开查看，计算正确性需结合结果核对。", "partial": "代码执行完成，部分工具未取得结果。"}.get(
            step.get("status"), "代码执行失败，错误已记录。",
        )
    return {
        "model": "模型调用未完成。",
        "select_tools": "已选择本轮所需工具并读取接口说明。",
        "read_context": "已读取当前案例的历史记录。",
        "context_budget": "当前问题和必要说明超过上下文预算。",
        "format_repair": "本次模型动作格式无效，已请求重新生成；本次动作未执行。",
        "citation_repair": "正在补齐资料引用并核对原文。",
        "knowledge_reuse": "相同检索已成功，复用已有资料并读取缺少的原文。",
        "finish": "已整理现有证据并完成回答。" if step.get("status") == "ok" else "已整理现有证据，分析尚未全部完成。",
        "stop": "本轮已停止，已有结果已保留。",
    }.get(action, "Agent 正在分析。")


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
        "ok": "已完成", "success": "已完成", "completed": "已完成",
        "partial": "部分完成", "cancelled": "已取消", "canceled": "已取消",
        "interrupted": "执行中断", "error": "未完成", "failed": "未完成",
        "failure": "未完成", "unavailable": "数据不可用", "blocked": "未完成",
        "clarification_needed": "等待补充信息", "unknown": "结果待确认",
    }
    return {"status": normalized, "label": labels.get(normalized, "结果待确认"),
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
