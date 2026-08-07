from __future__ import annotations

import json
import re
from typing import Any, Callable

from .contracts import ExecutionRequest, data_descs_from_context
from ..code_execution.client import SafeExecutionClient
from .runtime import TOOL_CLASSES
from .tool_docs import selected_api_reference


DIRECT_SYSTEM_PROMPT = """You are a SEVIR data-analysis code agent.
Generate a short Python program containing exactly one run() function.
Its arguments must exactly match the supplied signature and it must return the final result.
Do not import modules, access files, use network/system commands, or invent values.
Return JSON only:
{"plan":"one short sentence","code":"..."}
No chain-of-thought is needed. Use ds and only the supplied tool arguments.
"""


def _parse_program(content: str) -> dict[str, str]:
    text = str(content or "").strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and isinstance(value.get("code"), str):
                return {"plan": str(value.get("plan", "")), "code": value["code"]}
        except json.JSONDecodeError:
            continue
    code_match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_match:
        return {"plan": "Generated SEVIRWorld analysis", "code": code_match.group(1).strip()}
    raise ValueError("model did not return JSON with a code field")


def _compact(value: Any, limit: int = 60_000) -> str:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return encoded if len(encoded) <= limit else encoded[:limit] + "...[truncated]"


def _tool_runs(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": item.get("step"),
            "tool_name": item.get("api"),
            "tool_arguments": item.get("arguments", {}),
            "tool_result": item.get("result", {}),
        }
        for item in calls
    ]


def run_direct_agent(
    provider: dict[str, str],
    model: str,
    question: str,
    context: dict[str, Any],
    complete: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]],
    executor: SafeExecutionClient | None = None,
    max_attempts: int = 3,
    data_descs: list[dict[str, Any]] | None = None,
    selected_tools: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Generate, execute, repair, and synthesize a Direct-mode analysis.

    `complete` receives the provider/model and chat messages through a small
    callback supplied by the HTTP server. Execution observations, errors and
    repaired programs are retained; only short plans are exposed to the UI.
    """
    if not provider.get("api_key"):
        raise RuntimeError(f"{provider.get('label', 'model')} API key is not configured")
    executor = executor or SafeExecutionClient()
    request = ExecutionRequest.from_payload(
        data_descs if data_descs is not None else data_descs_from_context(context),
        selected_tools,
    )
    context_json = _compact(context, 8_000)
    signature = f"def run({', '.join(request.selected_tools)}):"
    api_reference = selected_api_reference(request.selected_tools, TOOL_CLASSES)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                DIRECT_SYSTEM_PROMPT
                + f"\nRequired signature:\n{signature}\n\nSelected API:\n"
                + api_reference
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"data_descs JSON: {_compact(request.data_descs, 8_000)}\n"
                f"Request settings JSON: {context_json}"
            ),
        },
    ]
    trace: list[dict[str, Any]] = []
    execution: dict[str, Any] | None = None
    program: dict[str, str] | None = None
    for attempt in range(1, max_attempts + 1):
        response = complete(provider, model, messages)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            program = _parse_program(content)
            execution = executor.execute(
                program["code"],
                context,
                data_descs=list(request.data_descs),
                selected_tools=request.selected_tools,
            )
        except Exception as error:
            execution = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            program = {"plan": "Program parsing failed", "code": str(content)}
        trace.append(
            {
                "attempt": attempt,
                "plan": program.get("plan", ""),
                "code": program.get("code", ""),
                "status": execution.get("status"),
                "error": execution.get("error"),
                "observation": execution.get("result") if execution.get("status") == "ok" else None,
                "calls": execution.get("calls", []),
                "elapsed_ms": execution.get("elapsed_ms"),
                "execution_id": execution.get("execution_id"),
            }
        )
        if execution.get("status") == "ok":
            break
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Execution failed. Repair the program and return JSON only. "
                        f"Error: {_compact(execution.get('error', 'unknown'), 4_000)}"
                    ),
                },
            ]
        )
    if not execution or execution.get("status") != "ok":
        raise RuntimeError(
            f"Direct code execution failed after {max_attempts} attempts: "
            f"{(execution or {}).get('error', 'unknown error')}"
        )

    primary_result = execution.get("primary_result") or execution.get("result") or {}
    calls = execution.get("calls", [])
    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "Summarize the executed SEVIRWorld result in concise Chinese. "
                "Use only the final code result JSON; mention only limits that still apply to "
                "that final result. A limitation inside an earlier nested tool result can be "
                "superseded by a later successful tool result. Do not invent values or claim "
                "that a tool was not run when its successful output is present."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nFinal executed result:\n{_compact(execution.get('result'))}",
        },
    ]
    synthesis = complete(provider, model, synthesis_messages)
    answer = synthesis.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not answer:
        answer = "代码已执行成功；请查看下方执行结果和图表。"
    return {
        "status": "ok",
        "answer": answer,
        "tool_name": "zephyrus_direct",
        "tool_arguments": {"attempts": len(trace)},
        "tool_result": primary_result,
        "tool_runs": _tool_runs(calls),
        "direct_trace": trace,
        "code_result": execution.get("result"),
        "route_mode": "zephyrus_direct",
        "artifacts": [
            item.get("tool_result")
            for item in _tool_runs(calls)
            if item.get("tool_result", {}).get("task") == "sevir_chart"
        ],
        "execution_limits": execution.get("limits", {}),
        "execution_id": execution.get("execution_id"),
        "data_descs": list(request.data_descs),
        "selected_tools": list(request.selected_tools),
    }
