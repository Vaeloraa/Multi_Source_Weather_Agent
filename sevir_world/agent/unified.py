"""Run one execute-observe-refine Agent with bounded execution and cancellation."""
from __future__ import annotations

import time
import re
import json
from dataclasses import replace
from typing import Any, Callable

from .contracts import ExecutionRequest, ExecutionRuntime
from .conversation import append_observation, initial_messages, parse_action, select_tool_contracts
from .context import ContextManager, ContextBudgetExceeded
from .tool_docs import tool_contract_archive
from .knowledge import repeated_search
from ..config import configured_agent_execution_timeout_seconds
from ..code_execution.client import ExecutionClient
from ..errors import error_record, failure_guidance
from ..serialization import SummaryLimits, serialize_value


SOLUTION_TIME_RESERVE_SECONDS = 20.0


class AgentCancelled(RuntimeError):
    """The caller is no longer waiting for this Agent run."""


def _execution_feedback(observation: dict[str, Any], previous: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose bounded execution facts, without inferring an analytical decision."""
    def identity(call: dict[str, Any]) -> str:
        return json.dumps([call.get("api"), call.get("arguments", {})], sort_keys=True, default=str)

    seen = {identity(call) for item in previous for call in item.get("calls", [])
            if call.get("status") == "ok"}
    repeated = []
    for call in observation.get("calls", []):
        key = identity(call)
        if call.get("status") == "ok":
            if key in seen:
                repeated.append(call)
            seen.add(key)
    feedback = {
        "result_preview": serialize_value(observation.get("result"), summary=SummaryLimits(
            max_characters=6000, max_depth=5, max_mapping_items=24,
            max_sequence_items=6, max_string_characters=500,
        )),
        "repeated_call_count": len(repeated),
        "repeated_cached_count": sum(call.get("cache_hit") is True for call in repeated),
        "repeated_apis": sorted({call["api"] for call in repeated}),
    }
    repeated_searches = [call for call in repeated if call.get("api") == "knowledge.search"]
    if repeated_searches:
        def source_identity(record):
            return json.dumps([record.get("chunk_id"), record.get("doc_id"), record.get("revision"),
                               record.get("text")], ensure_ascii=False)

        prior_sources = {source_identity(record) for item in previous for call in item.get("calls", [])
                         for record in call.get("knowledge_evidence", [])}
        new_sources = {source_identity(record) for call in repeated_searches
                       for record in call.get("knowledge_evidence", [])} - prior_sources
        feedback["knowledge_recovery"] = {
            "new_passage_count": len(new_sources),
            "next_action": "Read the retained passages or the per-source knowledge_sources archives with "
                           "read_context; follow next_offset. Return a cited solution when sufficient. "
                           "Search again only for a specific missing detail using changed parameters.",
        }
    return feedback


def run_unified_agent(
    provider: dict[str, str],
    model: str,
    question: str,
    complete: Callable[[dict[str, str], str, list[dict[str, str]]], dict[str, Any]],
    *,
    execution_request: ExecutionRequest,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    executor: ExecutionClient | None = None,
    max_steps: int = 8,
    max_executions: int = 5,
    max_error_retries: int = 3,
    runtime: ExecutionRuntime | None = None,
    summarize: Callable[[list[dict[str, str]]], str] | None = None,
) -> dict[str, Any]:
    """Run a prepared request, closing the Worker only when created here."""
    active_executor = executor if executor is not None else ExecutionClient(
        timeout_seconds=configured_agent_execution_timeout_seconds(), memory_limit_mb=4096,
    )
    try:
        return _run_session(
            provider, model, question, complete, execution_request,
            active_executor, runtime or ExecutionRuntime(), progress_callback,
            max(1, max_steps), max(1, max_executions), max(1, max_error_retries),
            summarize,
        )
    finally:
        if executor is None:
            active_executor.close()


def _run_session(
    provider: dict[str, str], model: str, question: str,
    complete: Callable, request: ExecutionRequest, executor: ExecutionClient,
    runtime: ExecutionRuntime, progress_callback: Callable | None,
    max_steps: int, max_executions: int, max_error_retries: int,
    summarize: Callable[[list[dict[str, str]]], str] | None,
) -> dict[str, Any]:
    messages = initial_messages(question, request)
    context = ContextManager(request.conversation_context, evidence=request.evidence_state,
                             summarize=summarize, cancelled=runtime.is_cancelled)
    available_tools = request.selected_tools
    active_tools = ("ds",)
    # Common-environment-only requests have no capability choice to make.
    selection_made = available_tools == ("ds",)
    request = replace(request, active_tools=active_tools)
    trace: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    execution_attempts = 0
    model_calls = 0
    consecutive_errors = 0
    format_errors = 0
    finalization_reason: str | None = None
    citation_repair_requested = False
    knowledge_rejections = 0

    def publish(event: str, **payload: Any) -> None:
        if progress_callback is not None:
            try:
                progress_callback({"event": event, **payload})
            except Exception:
                # A disconnected progress subscriber must not discard evidence.
                pass

    def record(action: str, status: str, **facts: Any) -> None:
        item = {"attempt": model_calls, "step": len(trace) + 1,
                "action": action, "status": status, **facts}
        trace.append(item)
        publish("step_completed", step=item)

    def finish(status: str, reason: str, answer: str = "", **extra: Any) -> dict[str, Any]:
        calls = [call for observation in observations for call in observation.get("calls", [])]
        from ..tools.knowledge.citations import resolve_citations
        answer, citations, citation_errors = resolve_citations(answer, calls)
        if citation_errors:
            status = "partial"
        return {
            "status": status,
            "answer": answer or "本轮未能完成完整分析，已保留取得的结果；缺失部分尚无可靠结论。",
            "route_mode": "unified_agent",
            "termination_reason": reason,
            "available_tools": list(available_tools),
            "selected_tools": list(active_tools) if selection_made else [],
            "generate_visual": request.generate_visual,
            "agent_trace": trace,
            "observations": observations,
            "tool_runs": calls,
            "knowledge_citations": citations,
            "citation_errors": citation_errors,
            "artifacts": list(artifacts.values()),
            "artifact_count": len(artifacts),
            "completion_requirements": {
                "visual_required": request.generate_visual,
                "visual_generated": bool(artifacts),
                "missing": ["visual"] if request.generate_visual and not artifacts else [],
            },
            "model_calls": model_calls,
            "context_summary": context.snapshot(),
            "context_archive": context.execution_archive(),
            "context_management": context.report(),
            "code_execution_attempt_count": execution_attempts,
            "code_execution_count": sum(
                item.get("postprocessing_succeeded", False)
                for item in observations
            ),
            "evidence_execution_count": sum(item.get("evidence_acquired", False) for item in observations),
            "evidence_tool_call_count": sum(item.get("evidence_tool_call_count", 0) for item in observations),
            "postprocessing_success_count": sum(item.get("postprocessing_succeeded", False) for item in observations),
            "tool_cost_summary": (observations[-1].get("tool_cost_summary") or {}) if observations else {},
            **extra,
        }

    for step in range(1, max_steps + 1):
        if runtime.is_cancelled():
            record("stop", "cancelled")
            return finish("cancelled", "client_cancelled", "请求已取消。")
        remaining = runtime.remaining_seconds()
        if remaining is not None and remaining <= 0:
            record("stop", "partial")
            return finish("partial", "execution_timeout")
        # Once finalization starts it never switches back to execution.
        if finalization_reason is None:
            if remaining is not None and remaining < SOLUTION_TIME_RESERVE_SECONDS:
                finalization_reason = "analysis_budget_exhausted"
            elif execution_attempts >= max_executions:
                finalization_reason = "analysis_budget_exhausted"
            elif observations and (observations[-1].get("tool_cost_summary") or {}).get("exhausted"):
                finalization_reason = "tool_cost_budget_exhausted"
            elif step == max_steps:
                finalization_reason = "max_steps_without_solution"
            if finalization_reason:
                messages.append({"role": "user", "content": (
                    "No further execution is allowed. Return one <solution> using existing "
                    "evidence; state any unfinished work or unavailable figure."
                )})
        started = time.perf_counter()
        publish("activity", attempt=step, action="context", status="running", model_calls=model_calls)
        try:
            model_messages = context.prepare(messages)
            model_calls += 1
            started = time.perf_counter()
            publish("activity", attempt=step, action="model", status="running", model_calls=model_calls)
            response = complete(provider, model, model_messages)
            choice = response["choices"][0]
        except ContextBudgetExceeded as error:
            checks = context.report().get("checks", [])
            budget = checks[-1] if checks else {}
            summary = (
                f"上下文压缩后估算输入仍为 {budget['after_estimated_tokens']:,} tokens，"
                f"超过本轮输入预算 {budget['input_limit']:,} tokens，未发起下一次模型调用。"
                if "after_estimated_tokens" in budget and "input_limit" in budget
                else "上下文超过输入预算，未发起下一次模型调用。"
            )
            record("context_budget", "partial", error=str(error), summary=summary,
                   elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            return finish("partial", "context_budget_exceeded", "本轮上下文暂时无法在预算内继续，已保留取得的记录；尚未形成完整结论。")
        except AgentCancelled:
            record("model", "cancelled", elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            return finish("cancelled", "client_cancelled", "请求已取消。")
        except Exception as error:
            record("model", "error", error_type=type(error).__name__, error=str(error),
                   elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            details = error_record(error, stage="model")
            failure_class = details["failure_class"]
            if failure_class == "unknown_error":
                failure_class = "model_transport_failure"
            message, next_action = failure_guidance(failure_class)
            if details.get("code") == "AllocationQuota.FreeTierOnly":
                message = "当前模型的免费额度已用尽，服务商的“仅使用免费额度”设置拒绝了请求。"
                next_action = "请检查模型服务控制台中的额度及“仅使用免费额度”设置后重试。"
            answer = message + (" 本轮已有执行记录已保留，但尚未生成完整回答。" if observations
                                else " 尚未取得模型回答。")
            return finish("partial", "model_transport_failure", answer, error=details,
                          failure_class=failure_class, failure_message=message, next_action=next_action)
        model_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if runtime.is_cancelled():
            record("stop", "cancelled")
            return finish("cancelled", "client_cancelled", "请求已取消。")
        try:
            message = choice["message"]
            action, body = parse_action(message.get("content") or "",
                                        finish_reason=choice.get("finish_reason", ""),
                                        tool_calls=message.get("tool_calls"))
        except ValueError as error:
            format_errors += 1
            record("format_repair", "error", error=str(error), elapsed_ms=model_elapsed_ms)
            if format_errors >= 2 or finalization_reason:
                return finish("partial", "model_protocol_error")
            messages.append({"role": "user", "content": (
                f"Nothing was executed: {error}. Return one complete select_tools, execute, read_context or solution block."
            )})
            continue

        if action == "solution":
            knowledge_evidence = [item for observation in observations for call in observation.get("calls", [])
                                  for item in call.get("knowledge_evidence", [])]
            if (knowledge_evidence and not re.search(r"\[K\d+\]|/api/knowledge/chunks/\d+", body)
                    and not citation_repair_requested and step < max_steps and not finalization_reason):
                citation_repair_requested = True
                record("citation_repair", "partial", elapsed_ms=model_elapsed_ms)
                messages.extend([
                    {"role": "assistant", "content": f"<solution>{body}</solution>"},
                    {"role": "user", "content": "Your document answer has no recorded source citations. "
                     "Return a concise <solution> citing the exact [K123] markers for each supported factual claim. "
                     "Recheck the original passages and remove guessed definitions, units or other unsupported details. "
                     "If the passages do not establish the requested detail, explicitly say it is unavailable; "
                     "do not cite background as proof of the missing detail."},
                ])
                continue
            # Completion is the Agent's decision. Worker status describes the
            # latest execution; arbitrary business result['status'] is not a protocol.
            incomplete = bool(observations and observations[-1]["status"] != "ok")
            missing_visual = request.generate_visual and not artifacts
            status = "partial" if incomplete or missing_visual else "ok"
            reason = "missing_required_visual" if missing_visual else "analysis_incomplete" if incomplete else "agent_completed"
            if finalization_reason == "model_protocol_error":
                status, reason = "partial", finalization_reason
            record("finish", status, elapsed_ms=model_elapsed_ms)
            return finish(status, reason, body)

        if action == "read_context":
            try:
                page = context.read(body)
            except (ValueError, TypeError, KeyError) as error:
                page = {"error": str(error)}
            messages.extend([
                {"role": "assistant", "content": f"<read_context>{body}</read_context>"},
                {"role": "user", "content": "<context_page>\n" + json.dumps(page, ensure_ascii=False) + "\n</context_page>"},
            ])
            record("read_context", "error" if "error" in page else "ok")
            continue

        remaining = runtime.remaining_seconds()
        if finalization_reason or (remaining is not None and remaining < SOLUTION_TIME_RESERVE_SECONDS):
            if finalization_reason:
                record("stop", "partial", elapsed_ms=model_elapsed_ms)
                return finish("partial", finalization_reason)
            finalization_reason = "analysis_budget_exhausted"
            messages.append({"role": "user", "content": (
                "The proposed program was not executed: time is reserved for the answer. "
                "Return one <solution> from existing evidence and state what is unfinished."
            )})
            continue

        if action == "select_tools":
            try:
                chosen, contracts = select_tool_contracts(
                    body, available_tools, active_tools, first=not selection_made,
                )
            except ValueError as error:
                format_errors += 1
                record("format_repair", "error", error=str(error), elapsed_ms=model_elapsed_ms)
                if format_errors >= 2:
                    return finish("partial", "model_protocol_error")
                messages.extend([
                    {"role": "assistant", "content": f"<select_tools>{body}</select_tools>"},
                    {"role": "user", "content": f"No tools were activated: {error}. Select from the catalog."},
                ])
                continue
            context.register_tool_contracts(tool_contract_archive(tuple(name for name in chosen if name not in active_tools)))
            active_tools = chosen
            selection_made = True
            request = replace(request, active_tools=active_tools)
            messages.extend([
                {"role": "assistant", "content": f"<select_tools>{body}</select_tools>"},
                {"role": "user", "content": (
                    f"Active capabilities: {', '.join(active_tools)}. "
                    "Only these tools are available to Python. Read the interfaces below, "
                    "then execute or answer. Select additional tools only if needed.\n\n" + contracts
                )},
            ])
            record("select_tools", "ok", selected_tools=list(active_tools), elapsed_ms=model_elapsed_ms,
                   summary="已选择分析所需工具：" + "、".join(active_tools) + "。")
            continue

        if not selection_made:
            format_errors += 1
            record("format_repair", "error", error="select tools before execution", elapsed_ms=model_elapsed_ms)
            if format_errors >= 2:
                return finish("partial", "model_protocol_error")
            messages.append({"role": "user", "content": (
                "Nothing was executed. First return <select_tools>[...]</select_tools> "
                "using names from the catalog, then read the supplied interfaces before writing Python."
            )})
            continue

        prior_search = repeated_search(body, observations)
        if prior_search is not None:
            knowledge_rejections += 1
            sources = context.knowledge_receipt([prior_search], context.policy.result_tokens)
            messages[:] = [item for item in messages if not item["content"].startswith("<knowledge_reuse>")]
            messages.append({"role": "user", "content": "<knowledge_reuse>\n"
                "Nothing was executed: this exact search already succeeded. Its evidence is available below. "
                "Read these source archives with read_context, following next_offset for full text, "
                "or return a cited solution if sufficient. Do not repeat the query. "
                "A distinct query is allowed when you identify a missing detail.\n"
                + json.dumps(sources, ensure_ascii=False) + "\n</knowledge_reuse>"})
            record("knowledge_reuse", "ok", execution_started=False,
                   arguments=prior_search.get("arguments"), elapsed_ms=model_elapsed_ms,
                   summary="相同检索已经成功，已提供原文摘录和归档入口，未重复执行。")
            if knowledge_rejections >= 2:
                finalization_reason = "analysis_incomplete"
                messages.append({"role": "user", "content":
                    "Repeated unchanged retrieval ended execution. Return a cited solution from the existing "
                    "sources and explicitly state any unanswered detail. No further execution is allowed."})
            continue
        knowledge_rejections = 0
        execution_started = time.perf_counter()
        publish("activity", attempt=step, action="execute", status="running",
                model_calls=model_calls, execution_attempts=execution_attempts + 1)
        observation = executor.execute(body, execution_request=request, runtime=runtime)
        if observation.get("preflight", {}).get("execution_started") is not False:
            execution_attempts += 1
        observation["execution_feedback"] = _execution_feedback(observation, observations)
        observations.append(observation)
        execution_elapsed_ms = round((time.perf_counter() - execution_started) * 1000, 2)
        if observation.get("postprocessing_succeeded"):
            for artifact in observation.get("generated_artifacts", []):
                artifacts[artifact["url"]] = artifact
        failed = observation["status"] != "ok"
        consecutive_errors = consecutive_errors + 1 if failed else 0
        record(
            "execute", observation["status"], code=body,
            elapsed_ms=execution_elapsed_ms, execution_elapsed_ms=execution_elapsed_ms,
            model_elapsed_ms=model_elapsed_ms,
            total_elapsed_ms=round(model_elapsed_ms + execution_elapsed_ms, 2),
            **{key: observation[key] for key in (
                "calls", "tool_cost_summary", "evidence_acquired", "postprocessing_succeeded", "error_type", "error",
                "execution_feedback",
            ) if key in observation},
        )
        if observation.get("error_type") == "ExecutionCancelled" or runtime.is_cancelled():
            record("stop", "cancelled")
            return finish("cancelled", "client_cancelled", "请求已取消。")
        if consecutive_errors >= max_error_retries:
            finalization_reason = "analysis_incomplete"
            messages.append({"role": "user", "content": (
                "Repeated execution errors ended execution. Return one <solution> with the "
                "available evidence and limitations; do not run more code."
            )})
        append_observation(messages, body, observation)

    record("stop", "partial")
    return finish("partial", finalization_reason or "max_steps_without_solution")
