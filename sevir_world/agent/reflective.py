from __future__ import annotations

import json
import re
from typing import Any, Callable

from .contracts import ExecutionRequest, data_descs_from_context
from .runtime import TOOL_CLASSES
from .tool_docs import selected_api_reference
from ..code_execution.client import SafeExecutionClient


REFLECTIVE_SYSTEM_PROMPT = """You are a reflective SEVIR weather-analysis code agent.
Use an execute-observe-refine loop. At each turn, return exactly one JSON object:
1. Execute one analysis step:
   {"action":"execute","plan":"one short sentence","code":"def run(...): ..."}
2. Finish after sufficient successful observations:
   {"action":"finish","answer":"concise Chinese answer grounded in observations"}

Do not provide chain-of-thought. The plan is only a short action summary.
Every code block must contain exactly one run() function with the supplied signature.
Do not import modules, access files, use network/system commands, or invent values.
After an observation, either run a genuinely useful next analysis or finish. Do not
repeat an equivalent call. You must successfully execute code at least once before
finishing. Mention applicable scientific limitations in the final answer.
"""


def _compact(value: Any, limit: int = 60_000) -> str:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return encoded if len(encoded) <= limit else encoded[:limit] + "...[truncated]"


def _parse_action(content: str) -> dict[str, str]:
    """Parse the bounded action protocol, with Zephyrus tags as a fallback."""
    text = str(content or "").strip()
    candidates = [text]
    candidates.extend(
        re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    )
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        action = str(value.get("action", "")).lower()
        if action == "execute" and isinstance(value.get("code"), str):
            return {
                "action": "execute",
                "plan": str(value.get("plan", "")),
                "code": value["code"],
            }
        if action == "finish" and isinstance(value.get("answer"), str):
            return {"action": "finish", "answer": value["answer"].strip()}

    execute_match = re.search(r"<execute>(.*?)</execute>", text, flags=re.DOTALL | re.IGNORECASE)
    if execute_match:
        return {
            "action": "execute",
            "plan": "Execute the next SEVIRWorld analysis",
            "code": execute_match.group(1).strip(),
        }
    solution_match = re.search(r"<solution>(.*?)</solution>", text, flags=re.DOTALL | re.IGNORECASE)
    if solution_match:
        return {"action": "finish", "answer": solution_match.group(1).strip()}
    raise ValueError("model did not return a valid execute or finish action")


def _tool_runs(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "tool_name": item.get("api"),
            "tool_arguments": item.get("arguments", {}),
            "tool_result": item.get("result", {}),
        }
        for index, item in enumerate(calls, start=1)
    ]


def run_reflective_agent(
    provider: dict[str, str],
    model: str,
    question: str,
    context: dict[str, Any],
    complete: Callable[[dict[str, Any], str, list[dict[str, Any]]], dict[str, Any]],
    executor: SafeExecutionClient | None = None,
    max_steps: int = 8,
    max_error_retries: int = 3,
    data_descs: list[dict[str, Any]] | None = None,
    selected_tools: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Execute a bounded Zephyrus-style execute-observe-refine interaction."""
    if not provider.get("api_key"):
        raise RuntimeError(f"{provider.get('label', 'model')} API key is not configured")
    if max_steps < 2:
        raise ValueError("max_steps must allow at least one execution and one finish step")

    executor = executor or SafeExecutionClient()
    request = ExecutionRequest.from_payload(
        data_descs if data_descs is not None else data_descs_from_context(context),
        selected_tools,
    )
    signature = f"def run({', '.join(request.selected_tools)}):"
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                REFLECTIVE_SYSTEM_PROMPT
                + f"\nRequired signature:\n{signature}\n\nSelected API:\n"
                + selected_api_reference(request.selected_tools, TOOL_CLASSES)
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"data_descs JSON: {_compact(request.data_descs, 8_000)}\n"
                f"Request settings JSON: {_compact(context, 8_000)}\n"
                "Begin by executing the minimum analysis needed to answer the question."
            ),
        },
    ]

    trace: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    successful_executions: list[dict[str, Any]] = []
    consecutive_errors = 0
    answer = ""

    for step in range(1, max_steps + 1):
        response = complete(provider, model, messages)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            action = _parse_action(content)
        except ValueError as error:
            consecutive_errors += 1
            trace.append(
                {
                    "attempt": step,
                    "action": "protocol_error",
                    "plan": "Invalid agent action",
                    "code": "",
                    "status": "error",
                    "error": str(error),
                }
            )
            if consecutive_errors >= max_error_retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": str(content)},
                    {
                        "role": "user",
                        "content": (
                            "Invalid response protocol. Return one JSON object with action=execute "
                            "and code, or action=finish and answer."
                        ),
                    },
                ]
            )
            continue

        if action["action"] == "finish":
            if not successful_executions:
                consecutive_errors += 1
                trace.append(
                    {
                        "attempt": step,
                        "action": "finish",
                        "plan": "Finish requested before observation",
                        "code": "",
                        "status": "rejected",
                        "error": "at least one successful execution is required",
                    }
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": str(content)},
                        {
                            "role": "user",
                            "content": (
                                "You cannot finish before a successful observation. "
                                "Return an execute action using the supplied API."
                            ),
                        },
                    ]
                )
                if consecutive_errors >= max_error_retries:
                    break
                continue
            answer = action["answer"]
            trace.append(
                {
                    "attempt": step,
                    "action": "finish",
                    "plan": "Answer from accumulated observations",
                    "code": "",
                    "status": "ok",
                    "error": None,
                }
            )
            break

        execution = executor.execute(
            action["code"],
            context,
            data_descs=list(request.data_descs),
            selected_tools=request.selected_tools,
        )
        trace.append(
            {
                "attempt": step,
                "action": "execute",
                "plan": action.get("plan", ""),
                "code": action["code"],
                "status": execution.get("status"),
                "error": execution.get("error"),
                "observation": execution.get("result") if execution.get("status") == "ok" else None,
                "calls": execution.get("calls", []),
                "elapsed_ms": execution.get("elapsed_ms"),
                "execution_id": execution.get("execution_id"),
            }
        )
        messages.append({"role": "assistant", "content": str(content)})
        if execution.get("status") == "ok":
            consecutive_errors = 0
            successful_executions.append(execution)
            all_calls.extend(execution.get("calls", []))
            observation = {
                "step": step,
                "result": execution.get("result"),
                "tool_calls": execution.get("calls", []),
            }
            observations.append(observation)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "<observation>"
                        + _compact(observation)
                        + "</observation>\n"
                        "Decide whether another distinct analysis is needed. Return exactly one "
                        "execute or finish JSON action."
                    ),
                }
            )
        else:
            consecutive_errors += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "<observation>Code execution failed: "
                        + _compact(execution.get("error", "unknown"), 4_000)
                        + "</observation>\nRepair the code and return an execute action."
                    ),
                }
            )
            if consecutive_errors >= max_error_retries:
                break

    if not successful_executions:
        last_error = trace[-1].get("error", "no successful observation") if trace else "no agent action"
        raise RuntimeError(f"Reflective agent failed without a successful observation: {last_error}")

    if not answer:
        fallback = complete(
            provider,
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Write a concise Chinese answer using only the supplied successful "
                        "SEVIRWorld observations. Do not invent values."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nObservations: {_compact(observations)}",
                },
            ],
        )
        answer = fallback.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not answer:
        answer = "分析步骤已执行成功，请查看执行观察和图表。"

    final_execution = successful_executions[-1]
    primary_result = final_execution.get("primary_result") or final_execution.get("result") or {}
    tool_runs = _tool_runs(all_calls)
    return {
        "status": "ok",
        "answer": answer,
        "tool_name": "zephyrus_reflective",
        "tool_arguments": {
            "agent_steps": len(trace),
            "successful_executions": len(successful_executions),
        },
        "tool_result": primary_result,
        "tool_runs": tool_runs,
        "reflective_trace": trace,
        "direct_trace": trace,
        "observations": observations,
        "code_result": final_execution.get("result"),
        "route_mode": "zephyrus_reflective",
        "artifacts": [
            item.get("tool_result")
            for item in tool_runs
            if item.get("tool_result", {}).get("task") == "sevir_chart"
        ],
        "execution_limits": final_execution.get("limits", {}),
        "execution_ids": [item.get("execution_id") for item in successful_executions],
        "data_descs": list(request.data_descs),
        "selected_tools": list(request.selected_tools),
    }
