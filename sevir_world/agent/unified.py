from __future__ import annotations

"""One Zephyrus-style execute-observe-refine Agent.

The same Agent may finish immediately, execute once, or continue after an
observation according to the unresolved user goal.
"""

import json
import re
import time
from typing import Any, Callable

from .contracts import ExecutionRequest
from .tool_docs import selected_api_reference
from ..code_execution.client import ExecutionClient


SYSTEM_PROMPT = """You are an analysis Agent that can write and execute Python against
request-bound data and tools.
You have an interactive standard-Python execution environment with real request-bound
data and tools.

At every turn, do exactly one of these:
1. Return one complete Python program inside <execute>...</execute>. The selected
   data and tools are already available as variables; assign compact output to result.
2. Return the final user-facing answer inside <solution>...</solution>.

The available capabilities do not prescribe an analysis plan. You decide which data,
tools, calculations, comparisons and figures answer the user's actual question. Inspect
the documented signatures, return types and examples instead of guessing fields. Use execution when an answer
depends on request-bound facts. After an observation, either
finish or execute a genuinely useful next analysis. Do not repeat a successful call
without a new reason.

The final answer must directly answer the user in clear Chinese. Do not show raw JSON,
Python, schemas, retries or internal protocol details. Embed requested figures with their
returned /api/artifacts/... URLs.
"""


class AgentCancelled(RuntimeError):
    """Raised when the HTTP client is no longer waiting for this Agent run."""


def _compact(value: Any, limit: int = 30_000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f" ...[truncated {len(text) - limit} characters]"


def _request_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """Expose already-bound request facts without leaking runtime internals.

    The executor has always received these values.  Hiding them from the Agent
    made it guess the event, cache policy and requested model, which in turn
    produced avoidable failed executions.  This is request state, not an
    analysis plan or a prewritten answer.
    """

    allowed = (
        "event_id",
        "data_source",
        "data_policy",
        "weather_model",
        "analysis_time_utc",
        "latitude",
        "longitude",
        "region",
        "forecast_leads_hours",
        "history_before_hours",
        "history_after_hours",
        "upload_id",
        "sequence_upload_id",
    )
    return {
        key: context[key]
        for key in allowed
        if key in context and context[key] not in (None, "", [], {})
    }


def _bound_asset_index(data_descs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Describe identities only; never open data or inspect cache contents."""

    assets: list[dict[str, Any]] = []
    for item in data_descs:
        source = str(item.get("source", ""))
        identity_key = "upload_id" if source == "upload" else "event_id"
        assets.append({
            "source": source,
            identity_key: item.get(identity_key),
            "representation": item.get("representation", "handle"),
        })
    return {
        "loading": "on_demand",
        "assets": assets,
        "details_expanded": False,
    }


def _parse_action(content: str) -> tuple[str, str]:
    text = str(content or "").strip()
    execute = re.search(r"<execute>(.*?)</execute>", text, re.DOTALL | re.IGNORECASE)
    solution = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if execute and solution:
        raise ValueError("return exactly one execute or solution block")
    if execute:
        return "execute", execute.group(1).strip()
    if solution:
        return "solution", solution.group(1).strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return "execute", fenced.group(1).strip()
    # A complete natural-language answer is already a valid terminal action.
    # Requiring a wrapper around it caused otherwise successful simple answers
    # to enter a format-repair turn and sometimes execute meaningless code.
    if text:
        return "solution", text
    raise ValueError("model returned an empty action")


def _artifact_records(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("image_url") or "")
            if re.fullmatch(r"/api/artifacts/[0-9a-f]{32}\.png", url) and url not in seen:
                seen.add(url)
                found.append(
                    {
                        "artifact_id": url.rsplit("/", 1)[-1].removesuffix(".png"),
                        "url": url,
                        "image_url": url,
                        "title": str(item.get("title") or "Agent生成图"),
                        "description": str(item.get("description") or "由本轮生成代码基于真实数据绘制。"),
                        "chart_type": str(item.get("chart_type") or "agent_generated"),
                        "status": "completed",
                    }
                )
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            for url in re.findall(r"/api/artifacts/[0-9a-f]{32}\.png", item):
                if url in seen:
                    continue
                seen.add(url)
                found.append(
                    {
                        "artifact_id": url.rsplit("/", 1)[-1].removesuffix(".png"),
                        "url": url,
                        "image_url": url,
                        "title": "Agent生成图",
                        "description": "由本轮生成代码基于真实数据绘制。",
                        "chart_type": "agent_generated",
                        "status": "completed",
                    }
                )

    visit(value)
    return found


def _tool_result_summary(call: dict[str, Any]) -> str:
    """Build a short factual trace sentence from the recorded tool result."""

    result = call.get("result")
    if not isinstance(result, dict):
        return f"返回 {call.get('return_type') or type(result).__name__}"
    facts: list[str] = []
    value_type = result.get("type") or call.get("return_type")
    if value_type:
        facts.append(f"返回 {value_type}")
    if result.get("event_id") or result.get("id"):
        facts.append(f"事件 {result.get('event_id') or result.get('id')}")
    if result.get("cache_hit") is True or call.get("cache_hit") is True:
        facts.append("命中已有数据缓存")
    for key, label in (
        ("history_frame_count", "历史观测帧"),
        ("forecast_frame_count", "预测帧"),
        ("frame_count", "数据帧"),
        ("ensemble_members", "集合成员"),
        ("ensemble_count", "集合成员"),
        ("artifact_count", "生成图表"),
    ):
        value = result.get(key)
        if value not in (None, "", [], {}):
            facts.append(f"{label} {value}")
    if result.get("image_url") or result.get("url"):
        facts.append("已生成可视化文件")
    return "；".join(facts[:5]) or "已返回可直接计算的 Python 对象"


def _run_unified_agent_session(
    provider: dict[str, str],
    model: str,
    question: str,
    context: dict[str, Any],
    complete: Callable[[dict[str, str], str, list[dict[str, str]]], dict[str, Any]],
    *,
    data_descs: list[dict[str, Any]] | None = None,
    selected_tools: list[str] | tuple[str, ...] | None = None,
    conversation_context: str = "",
    executor: ExecutionClient | None = None,
    max_steps: int = 8,
    max_error_retries: int = 3,
) -> dict[str, Any]:
    if executor is None:
        raise ValueError("executor is required for an Agent session")
    request = ExecutionRequest.from_payload(data_descs, selected_tools, context)
    dataset_description = _bound_asset_index(request.data_descs)
    reference = selected_api_reference(
        request.selected_tools,
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Request-bound context (facts already resolved by the application; do not guess or replace them):\n"
        f"{_compact(_request_context_for_prompt(context), limit=4_000)}\n\n"
        f"Injected variable names:\n{', '.join(request.selected_tools)}\n\n"
        f"Available runtime:\n{reference}"
    )
    if conversation_context:
        user_prompt += "\n\nRelevant conversation context:\n" + conversation_context

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    consecutive_errors = 0
    format_errors = 0
    successful_executions = 0
    model_calls = 0

    cancelled = context.get("_cancelled")
    is_cancelled = cancelled if callable(cancelled) else (lambda: False)

    def terminal_result(
        status: str,
        answer: str,
        termination_reason: str,
    ) -> dict[str, Any]:
        terminal = {
            "status": status,
            "answer": answer,
            "route_mode": "unified_agent",
            "termination_reason": termination_reason,
            "selected_tools": list(request.selected_tools),
            "agent_trace": trace,
            "observations": observations,
            "tool_runs": [
                call for item in observations for call in item.get("calls", [])
            ],
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "model_calls": model_calls,
            "code_execution_count": successful_executions,
            "dataset_description": dataset_description,
        }
        return terminal

    for step in range(1, max(2, int(max_steps)) + 1):
        if is_cancelled():
            trace.append({
                "attempt": step,
                "action": "stop",
                "status": "cancelled",
                "elapsed_ms": 0,
                "summary": "客户端已断开，停止剩余Agent工作",
            })
            return terminal_result(
                "cancelled", "请求已取消。", "client_cancelled"
            )
        call_started = time.perf_counter()
        model_calls += 1
        try:
            response = complete(provider, model, messages)
            if is_cancelled():
                raise AgentCancelled("client disconnected during model call")
        except AgentCancelled:
            model_elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
            trace.append({
                "attempt": step,
                "action": "model",
                "status": "cancelled",
                "elapsed_ms": model_elapsed_ms,
                "summary": "客户端已断开，停止剩余Agent工作",
            })
            return terminal_result(
                "cancelled", "请求已取消。", "client_cancelled"
            )
        except Exception as error:
            model_elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
            trace.append({
                "attempt": step,
                "action": "model",
                "status": "error",
                "elapsed_ms": model_elapsed_ms,
                "summary": f"模型调用失败：{type(error).__name__}",
                "error_type": type(error).__name__,
                "error": str(error)[:1200],
            })
            valid_observations = [
                item for item in observations if item.get("status") == "ok"
            ]
            if valid_observations:
                return terminal_result(
                    "partial",
                    "本轮已经获得并保留真实执行结果，但最终模型调用失败，未生成完整文字结论。请查看步骤和已生成图表。",
                    "model_failure_after_observation",
                )
            raise
        content = str(response.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        model_elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
        if not content:
            raise RuntimeError("language model returned an empty Agent action")
        try:
            action, body = _parse_action(content)
        except ValueError as error:
            format_errors += 1
            trace.append(
                {
                    "attempt": step,
                    "action": "format_repair",
                    "status": "error",
                    "elapsed_ms": model_elapsed_ms,
                    "summary": "模型输出缺少可执行动作或最终回答标签",
                }
            )
            if format_errors >= 2:
                raise RuntimeError(str(error)) from error
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Protocol only: return exactly one <execute>...</execute> or <solution>...</solution> block.",
                    },
                ]
            )
            continue

        if action == "solution":
            trace.append(
                {
                    "attempt": step,
                    "action": "finish",
                    "status": "ok",
                    "elapsed_ms": model_elapsed_ms,
                    "summary": "已根据现有证据完成回答",
                }
            )
            return terminal_result("ok", body, "agent_completed")

        execution_started = time.perf_counter()
        execution = executor.execute(
            body,
            context,
            data_descs=list(request.data_descs),
            selected_tools=request.selected_tools,
        )
        execution_elapsed_ms = round((time.perf_counter() - execution_started) * 1000, 2)
        call_summaries = [
            {
                "api": call.get("api"),
                "arguments": call.get("arguments", {}),
                "status": call.get("status"),
                "cache_hit": call.get("cache_hit", False),
                "error": call.get("error"),
                "return_type": call.get("return_type"),
                "result_summary": _tool_result_summary(call),
            }
            for call in execution.get("calls", [])
            if isinstance(call, dict)
        ]
        observation = {
            "status": execution.get("status"),
            "result": execution.get("result"),
            "stdout": execution.get("stdout", ""),
            "calls": call_summaries,
            "generated_artifacts": execution.get("generated_artifacts", []),
            "error_type": execution.get("error_type"),
            "error": execution.get("error"),
            "traceback": execution.get("traceback"),
            "error_location": execution.get("error_location"),
        }
        observations.append(observation)
        for artifact in _artifact_records(observation):
            if artifact["url"] not in {item["url"] for item in artifacts}:
                artifacts.append(artifact)

        if execution.get("status") == "ok":
            successful_executions += 1
            consecutive_errors = 0
            summary = "代码执行成功，已获得新的真实观测结果"
        else:
            consecutive_errors += 1
            summary = f"代码执行失败：{execution.get('error_type') or 'ExecutionError'}"
        trace.append(
            {
                "attempt": step,
                "action": "execute",
                "status": execution.get("status", "error"),
                "elapsed_ms": execution_elapsed_ms,
                "model_elapsed_ms": model_elapsed_ms,
                "summary": summary,
                "code": body,
                "calls": call_summaries,
                "error_type": execution.get("error_type"),
                "error": execution.get("error"),
            }
        )
        if execution.get("error_type") == "ExecutionCancelled" or is_cancelled():
            trace.append({
                "attempt": step,
                "action": "stop",
                "status": "cancelled",
                "elapsed_ms": 0,
                "summary": "客户端已断开，停止剩余Agent工作",
            })
            return terminal_result(
                "cancelled", "请求已取消。", "client_cancelled"
            )
        if consecutive_errors >= max(1, int(max_error_retries)):
            trace.append(
                {
                    "attempt": step,
                    "action": "stop",
                    "status": "error",
                    "elapsed_ms": 0,
                    "summary": f"连续{consecutive_errors}次执行失败，停止本轮",
                }
            )
            return terminal_result(
                "error",
                "本轮代码连续执行失败，完整步骤已保留供检查。",
                "consecutive_execution_errors",
            )
        messages.extend(
            [
                {"role": "assistant", "content": f"<execute>{body}</execute>"},
                {
                    "role": "user",
                    "content": (
                        "<observation>\n"
                        + _compact(observation)
                        + "\n</observation>\n"
                        "Decide whether the user's goal is complete. Return <solution> if it is; "
                        "otherwise execute one useful next analysis. If execution failed, use the "
                        "actual traceback to repair the program."
                    ),
                },
            ]
        )

    # Preserve the complete audit trail when the model fails to terminate.
    # Raising here used to erase the most valuable evidence for diagnosing a
    # wandering Agent and made an ordinary stopping failure look like an API or
    # executor failure to the UI.
    trace.append(
        {
            "attempt": max(2, int(max_steps)),
            "action": "stop",
            "status": "incomplete",
            "elapsed_ms": 0,
            "summary": f"Agent已执行{successful_executions}次，但未在轮次内形成最终回答",
        }
    )
    return terminal_result(
        "incomplete",
        "本轮已保留全部真实执行结果，但智能体没有在分析轮次内完成最终回答。",
        "max_steps_without_solution",
    )


def run_unified_agent(
    provider: dict[str, str],
    model: str,
    question: str,
    context: dict[str, Any],
    complete: Callable[[dict[str, str], str, list[dict[str, str]]], dict[str, Any]],
    *,
    data_descs: list[dict[str, Any]] | None = None,
    selected_tools: list[str] | tuple[str, ...] | None = None,
    conversation_context: str = "",
    executor: ExecutionClient | None = None,
    max_steps: int = 8,
    max_error_retries: int = 3,
) -> dict[str, Any]:
    """Run one Agent session and always terminate an internally owned Worker."""

    owned_executor = executor is None
    active_executor = executor or ExecutionClient(timeout_seconds=90, memory_limit_mb=4096)
    try:
        return _run_unified_agent_session(
            provider,
            model,
            question,
            context,
            complete,
            data_descs=data_descs,
            selected_tools=selected_tools,
            conversation_context=conversation_context,
            executor=active_executor,
            max_steps=max_steps,
            max_error_retries=max_error_retries,
        )
    finally:
        if owned_executor:
            active_executor.close()
