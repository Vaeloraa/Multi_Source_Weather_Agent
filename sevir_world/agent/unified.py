from __future__ import annotations

import os
"""One Zephyrus-style execute-observe-refine Agent.

The same Agent may finish immediately, execute once, or continue after an
observation according to the unresolved user goal.
"""

import ast
import json
import re
import time
from dataclasses import asdict
from typing import Any, Callable

from .contracts import CANONICAL_ARGUMENTS, ExecutionRequest
from .tool_docs import capability_index, compact_api_reference, selected_api_reference
from .visual_contract import visual_artifact_required
from ..code_execution.client import ExecutionClient


SYSTEM_PROMPT = """You are an analysis Agent that can write and execute Python against
request-bound data and tools.
You have an interactive standard-Python execution environment with real request-bound
data and tools.

At every turn, do exactly one of these:
1. Return one complete Python program inside <execute>...</execute>. The selected
   data and tools are already available as variables; assign compact output to result.
2. Return the final user-facing answer inside <solution>...</solution>.
3. Before first using a capability whose detailed contract is not visible, return
   <expand>sevir</expand> (or comma-separated capability names). This reveals its
   source-derived signatures, return fields and examples for the following turn.

The available capabilities do not prescribe an analysis plan. You decide which data,
tools, calculations, comparisons and figures answer the user's actual question. Inspect
the documented signatures, return types and examples instead of guessing fields. Use execution when an answer
depends on request-bound facts. After an observation, either
finish or execute a genuinely useful next analysis. Do not repeat a successful call
without a new reason. Gather closely related setup facts in one coherent execution;
do not spend separate turns only printing metadata that can be read together. When a
requested figure uses values computed in the same analysis, calculate and draw it in
the same execution instead of splitting calculation and plotting across turns: compute
the numbers and save the figure inside one single <execute> program. Only call the
capabilities the question actually needs; skip event or location lookups that do not
change the answer.

For a multi-request Study Case, prior factual evidence is available through the
read-only ``study`` object. Reuse that evidence instead of reconstructing it from
prior answer prose or repeating successful tools. When the current analysis makes a
decision that later requests must inherit, publish the small validated decision in
``result["study_update"]`` as documented by the runtime. This is part of the evidence
contract, not a narrative summary.

The final answer must directly answer the user in clear Chinese. Do not show raw JSON,
Python, schemas, retries or internal protocol details. Embed requested figures only with
/api/artifacts/... URLs returned by the runtime; never invent external image links.
Keep analytical claims faithful to the returned result. A comparison of change, trend or
evolution must measure at least two times for every compared group; a cross-sectional
difference at one time is not an evolution. Keep discovered, loadable, selected and
actually analysed sample counts distinct. If a requested comparison is empty or None,
compute it in another execution or state that it is unavailable instead of inferring it.
Create a figure only when the request-level visual artifact contract says it is required,
or when a visual materially improves an otherwise difficult relationship. Never create an
ornamental figure merely to satisfy a generic data-backed-answer rule. When a visual is
required, save it through SEVIR_ARTIFACT_DIR and cite the registered /api/artifacts/... URL.
Never invent values just to make a figure. Keep the final answer focused and concise —
normally within 300 Chinese characters — and free of filler.
"""

# A final synthesis needs room to generate the user-facing answer.  Once the
# remaining request budget drops below this reserve, the Agent stops starting
# new executions and spends what is left on producing the answer instead of
# dying mid-generation with evidence but no conclusion.
SOLUTION_TIME_RESERVE_SECONDS = 20.0
# A reflective audit spends one full model call and may trigger one repair, so
# it only runs while the remaining budget comfortably covers both.
REVIEW_TIME_FLOOR_SECONDS = 40.0
# When no figure has been saved yet, the final answer will be rejected without
# one, so the Agent keeps a smaller reserve that still allows a last-chance
# figure execution instead of forcing a guaranteed-invalid answer.
LAST_FIGURE_RESERVE_SECONDS = 14.0


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


def _request_context_for_prompt(request: ExecutionRequest) -> dict[str, Any]:
    """Expose the exact binding and data identities the runtime will receive.

    The executor has always received these values.  Hiding them from the Agent
    made it guess the event, cache policy and requested model, which in turn
    produced avoidable failed executions.  This is request state, not an
    analysis plan or a prewritten answer.
    """

    binding = request.binding
    visible = {
        "assets": asdict(binding.assets),
        "target": asdict(binding.target),
        "forecast": asdict(binding.forecast),
        "policy": asdict(binding.policy),
        "provenance": asdict(binding.provenance),
        "response": {
            "visual_artifact_required": request.visual_artifact_required,
        },
    }
    visible["data_assets"] = _bound_asset_index(request.data_descs)["assets"]
    return visible


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


def _parse_action(content: str, *, finish_reason: str = "") -> tuple[str, str]:
    """Parse exactly one complete Agent action from a model response.

    A text prompt is not a protocol guarantee.  Count complete and opening
    tags explicitly so a response containing several actions, or one complete
    action followed by a truncated action, can never be partially executed.
    """
    text = str(content or "").strip()
    if str(finish_reason or "").strip().lower() == "length":
        raise ValueError("model action was truncated by the output token limit")
    block_pattern = re.compile(
        r"<(execute|solution|expand)>(.*?)</\1>", re.DOTALL | re.IGNORECASE
    )
    blocks = list(block_pattern.finditer(text))
    openings = re.findall(r"<(?:execute|solution|expand)\b", text, re.IGNORECASE)
    if not blocks and len(openings) == 1 and str(finish_reason).lower() == "stop":
        opening = re.search(r"<(execute|solution|expand)>", text, re.IGNORECASE)
        if opening is not None:
            body = text[opening.end():].strip()
            if body:
                return opening.group(1).lower(), body
    if len(blocks) != 1 or len(openings) != 1:
        if openings:
            raise ValueError("return exactly one complete execute, solution, or expand block")
    if blocks:
        action = blocks[0].group(1).lower()
        body = blocks[0].group(2).strip()
        if not body:
            raise ValueError(f"model returned an empty {action} block")
        return action, body
    if openings:
        raise ValueError("model returned an incomplete protocol block")
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if len(fences) > 1:
        raise ValueError("return one Python action, not several fenced programs")
    if fences:
        return "execute", fences[0].strip()
    # A complete natural-language answer is already a valid terminal action.
    # Requiring a wrapper around it caused otherwise successful simple answers
    # to enter a format-repair turn and sometimes execute meaningless code.
    if text:
        return "solution", text
    raise ValueError("model returned an empty action")


def _capabilities_used_in_code(code: str) -> set[str]:
    """Return public capability variables read by one generated program."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    public = {"sevir", "geolocator", "cascast", "aurora"}
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in public
    }
def _action_message(content: str, action: str, body: str) -> str:
    """Preserve valid model reasoning while canonicalising its action block.

    The completion stop sequence normally removes the closing action tag.  The
    controller still needs a complete assistant message in the next turn, but
    rebuilding only the code block used to erase the model's analysis preamble
    and break reflective continuity.
    """

    opening = re.search(rf"<{re.escape(action)}>", content, re.IGNORECASE)
    preamble = content[: opening.start()].strip() if opening is not None else ""
    block = f"<{action}>{body}</{action}>"
    return f"{preamble}\n\n{block}" if preamble else block


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


def _prompt_value(value: Any, *, depth: int = 0) -> Any:
    """Bound nested execution values before they enter a follow-up prompt."""
    if depth > 6:
        return f"<{type(value).__name__}>"
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…[truncated]"
    if isinstance(value, dict):
        items = list(value.items())[:40]
        return {str(key): _prompt_value(item, depth=depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return {
                "type": type(value).__name__,
                "count": len(value),
                "first": _prompt_value(value[0], depth=depth + 1),
                "last": _prompt_value(value[-1], depth=depth + 1),
            }
        return [_prompt_value(item, depth=depth + 1) for item in value]
    return value


def _folded_observation(observation: dict[str, Any]) -> str:
    """Summarise an older observation so the prompt stops growing.

    Only the two newest observations stay verbatim in the conversation;
    earlier ones collapse to status, artifacts and errors.  The Worker
    namespace keeps every computed value alive, so folding prompt text never
    destroys data the model may still reuse.
    """

    lines = [
        "<observation-summary>",
        f"较早一次执行的观察已折叠：status={observation.get('status') or 'unknown'}",
    ]
    error_type = observation.get("error_type")
    error = observation.get("error")
    if error_type or error:
        lines.append(f"错误：{error_type or 'Error'}: {error}")
    artifact_urls = [
        str(item.get("url"))
        for item in observation.get("generated_artifacts", [])
        if isinstance(item, dict) and item.get("url")
    ]
    if artifact_urls:
        lines.append("产物：" + ", ".join(artifact_urls))
    lines.append(
        "该次执行的详细数值仍保留在 Worker 命名空间中（见最新观察的 available_names），"
        "需要时直接在后续代码里复用，无需重新调用接口。"
    )
    lines.append("</observation-summary>")
    return "\n".join(lines)


def _fold_capability_contracts(messages: list[dict[str, str]]) -> None:
    """Fold prose/examples but permanently retain signatures and return structure."""

    for message in messages:
        content = str(message.get("content", ""))
        match = re.match(
            r"<capability-contracts names=\"([^\"]+)\">",
            content,
        )
        if match:
            names = tuple(
                name.strip() for name in match.group(1).split(",") if name.strip()
            )
            message["content"] = (
                f"<capability-contract-summary names=\"{match.group(1)}\">\n"
                + compact_api_reference(names)
                + "\nLong explanations and examples were folded, but this core contract "
                "remains authoritative for every repair turn. Request <expand> only when "
                "the full source docstring or examples are genuinely needed.\n"
                "</capability-contract-summary>"
            )


def _observation_for_model(observation: dict[str, Any]) -> dict[str, Any]:
    """Pass evidence to the next turn without repeating transport noise."""
    return {
        "status": observation.get("status"),
        "result": _prompt_value(observation.get("result")),
        "error_type": observation.get("error_type"),
        "error": observation.get("error"),
        "error_location": observation.get("error_location"),
        "namespace_state": observation.get("namespace_state", {}),
        "failed_value_shapes": observation.get("failed_value_shapes", {}),
        # The Worker records the full traceback for the repair loop; passing
        # it through lets the model fix the exact failing line instead of
        # regenerating the whole program, which keeps repair turns small.
        "traceback": _compact(observation.get("traceback", ""), limit=2_000),
        "stdout": _compact(observation.get("stdout", ""), limit=1_200),
        "available_names": observation.get("available_names", []),
        "generated_artifacts": observation.get("generated_artifacts", []),
        "tool_call_summary": observation.get("tool_call_summary", {}),
        "calls": [
            {
                "api": call.get("api"),
                "status": call.get("status"),
                "return_type": call.get("return_type"),
                "result_summary": call.get("result_summary"),
                "error": call.get("error"),
            }
            for call in observation.get("calls", [])
            if isinstance(call, dict)
        ],
    }


def _solution_artifact_error(
    body: str,
    artifacts: list[dict[str, Any]],
    *,
    required: bool = False,
) -> str | None:
    """Reject invented or omitted figure links at the final evidence boundary."""

    raw_references = set(
        re.findall(r"/api/artifacts/[^\s)\]>'\"]+", body, re.IGNORECASE)
    )
    malformed = sorted(
        item
        for item in raw_references
        if not re.fullmatch(r"/api/artifacts/[0-9a-f]{32}\.png", item)
    )
    if malformed:
        return "solution referenced malformed or unregistered artifacts: " + ", ".join(malformed)
    referenced = set(raw_references)
    registered = {
        str(item.get("url"))
        for item in artifacts
        if re.fullmatch(r"/api/artifacts/[0-9a-f]{32}\.png", str(item.get("url") or ""))
    }
    if required and not registered:
        return "the user explicitly requested a visual artifact, but none was registered"
    unknown = sorted(referenced - registered)
    if unknown:
        return "solution referenced unregistered artifacts: " + ", ".join(unknown)
    if registered and not referenced:
        return "solution omitted all registered evidence figures"
    return None


def _run_unified_agent_session(
    provider: dict[str, str],
    model: str,
    question: str,
    context: dict[str, Any],
    complete: Callable[[dict[str, str], str, list[dict[str, str]]], dict[str, Any]],
    *,
    data_descs: list[dict[str, Any]] | None = None,
    selected_tools: list[str] | tuple[str, ...] | None = None,
    execution_request: ExecutionRequest | None = None,
    conversation_context: str = "",
    executor: ExecutionClient | None = None,
    max_steps: int = 8,
    max_executions: int = 5,
    max_error_retries: int = 3,
) -> dict[str, Any]:
    if executor is None:
        raise ValueError("executor is required for an Agent session")
    request = execution_request or ExecutionRequest.from_payload(
        data_descs,
        selected_tools,
        context,
        dict(context.get("_study_case") or {}),
    )
    dataset_description = _bound_asset_index(request.data_descs)
    visual_required = request.visual_artifact_required or (
        visual_artifact_required(question)
    )
    # Runtime always injects all four lazy capabilities. The first layer lists
    # all four consistently; routing hints never make an object undiscoverable.
    reference = capability_index(CANONICAL_ARGUMENTS)
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Request-bound context (facts already resolved by the application; do not guess or replace them):\n"
        f"{_compact(_request_context_for_prompt(request), limit=4_000)}\n\n"
        f"Injected variable names:\n{', '.join(CANONICAL_ARGUMENTS)}, study\n\n"
        f"Visual artifact contract:\nrequired={str(visual_required).lower()}. "
        + (
            "Create and cite at least one registered artifact that directly answers the request.\n\n"
            if visual_required
            else "A figure is optional; do not create one solely for acceptance.\n\n"
        )
        + f"Available runtime:\n{reference}"
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
    execution_attempts = 0
    evidence_execution_count = 0
    evidence_tool_call_count = 0
    postprocessing_success_count = 0
    latest_tool_cost_summary: dict[str, Any] = {}
    tool_cost_blocked = False
    expanded_capabilities: set[str] = set()
    revealed_capabilities: set[str] = set()
    model_calls = 0
    solution_validation_errors = 0
    solution_reviewed = False

    cancelled = context.get("_cancelled")
    is_cancelled = cancelled if callable(cancelled) else (lambda: False)

    def request_deadline() -> float | None:
        value = context.get("_deadline_monotonic")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

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
            "code_execution_attempt_count": execution_attempts,
            "evidence_execution_count": evidence_execution_count,
            "evidence_tool_call_count": evidence_tool_call_count,
            "postprocessing_success_count": postprocessing_success_count,
            "tool_cost_summary": latest_tool_cost_summary,
            "dataset_description": dataset_description,
        }
        return terminal

    # ``max_steps`` is the actual number of model turns, not an implicit
    # extra retry.  The old ``max(2, ...) + 1`` made a nominal two-turn
    # request execute three long-context calls and routinely consumed the
    # entire request budget before a solution could be written.
    for step in range(1, max(1, int(max_steps)) + 1):
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
            deadline = request_deadline()
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Agent model call exceeded the request deadline")
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
                item for item in observations if item.get("status") in {"ok", "partial"}
            ]
            model_failure = {
                "type": type(error).__name__,
                "message": str(error)[:1200],
            }
            if valid_observations:
                terminal = terminal_result(
                    "partial",
                    "本轮已经获得并保留真实执行结果，但最终模型调用失败，未生成完整文字结论。请查看步骤和已生成图表。",
                    "model_failure_after_observation",
                )
                terminal["failure_class"] = "model_transport_failure"
                terminal["error"] = model_failure
                return terminal
            terminal = terminal_result(
                "error",
                "固定模型调用失败；未继续重试，也未切换其他模型。",
                "model_transport_failure",
            )
            terminal["failure_class"] = "model_transport_failure"
            terminal["error"] = model_failure
            return terminal
        choice = (response.get("choices") or [{}])[0]
        content = str(choice.get("message", {}).get("content", "")).strip()
        finish_reason = str(choice.get("finish_reason") or "")
        model_elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
        if not content:
            raise RuntimeError("language model returned an empty Agent action")
        try:
            action, body = _parse_action(content, finish_reason=finish_reason)
        except ValueError as error:
            format_errors += 1
            trace.append(
                {
                    "attempt": step,
                    "action": "format_repair",
                    "status": "error",
                    "elapsed_ms": model_elapsed_ms,
                    "summary": f"模型动作协议无效：{error}",
                    "error": str(error),
                    "finish_reason": finish_reason or None,
                }
            )
            if format_errors >= 2:
                terminal = terminal_result(
                    "error",
                    "固定模型连续两次没有返回一个完整且唯一的Agent动作。",
                    "model_protocol_error",
                )
                terminal["failure_class"] = "model_protocol_error"
                terminal["error"] = {"type": "ModelProtocolError", "message": str(error)}
                return terminal
            # Do not feed an invalid, possibly truncated multi-action response
            # back into the conversation.  It anchors the model to continue a
            # speculative plan whose earlier steps were never executed.
            messages.append({
                "role": "user",
                "content": (
                    "Your previous action was invalid and nothing from it was executed: "
                    + str(error) + ". Return exactly one short, complete "
                    "<execute>...</execute>, <solution>...</solution>, or "
                    "<expand>capability</expand> block. "
                    "Do not continue the discarded response or include a future action."
                ),
            })
            continue

        if action == "expand":
            requested_names = tuple(dict.fromkeys(
                name.lower()
                for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body)
                if name.lower() in CANONICAL_ARGUMENTS and name.lower() != "ds"
            ))
            if not requested_names:
                messages.append({
                    "role": "user",
                    "content": (
                        "No valid capability was named. Return one <expand> block using "
                        "sevir, geolocator, cascast, or aurora."
                    ),
                })
                continue
            requested_unseen = tuple(
                name for name in requested_names if name not in expanded_capabilities
            )
            if not requested_unseen:
                messages.extend([
                    {"role": "assistant", "content": _action_message(content, action, body)},
                    {
                        "role": "user",
                        "content": (
                            "Those capability contracts are already visible in this turn. "
                            "Do not request them again; return one <execute> or <solution>."
                        ),
                    },
                ])
                continue
            # Progressive disclosure is per capability. Combining two or more
            # complete source docstrings recreates the old flat 15–20k prompt
            # and can consume the entire fixed-model request budget before any
            # tool is used. Reveal the first requested capability now; the
            # Agent may request another one on a later turn if its evidence is
            # actually needed.
            newly_expanded = requested_unseen[:1]
            deferred_names = requested_unseen[1:]
            expanded_capabilities.update(newly_expanded)
            revealed_capabilities.update(newly_expanded)
            details = selected_api_reference(newly_expanded, include_policy=True)
            trace.append({
                "attempt": step,
                "action": "expand",
                "status": "ok",
                "elapsed_ms": model_elapsed_ms,
                "summary": "已按需展开能力契约：" + ", ".join(newly_expanded),
                "capabilities": list(newly_expanded),
                "new_capabilities": list(newly_expanded),
            })
            messages.extend([
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        f'<capability-contracts names="{",".join(newly_expanded)}">\n'
                        + details
                        + "\n</capability-contracts>\n"
                        + (
                            "The other requested capability contracts were deferred to keep "
                            "progressive disclosure bounded; request one later only if needed.\n"
                            if deferred_names else ""
                        )
                        + "Use this public contract directly. Return exactly one next action."
                    ),
                },
            ])
            continue

        if action == "solution":
            artifact_error = _solution_artifact_error(
                body, artifacts, required=visual_required
            )
            if artifact_error:
                solution_validation_errors += 1
                trace.append(
                    {
                        "attempt": step,
                        "action": "solution_validation",
                        "status": "error",
                        "elapsed_ms": model_elapsed_ms,
                        "summary": "最终回答引用的图像与真实执行产物不一致",
                        "error": artifact_error,
                    }
                )
                if solution_validation_errors >= 2:
                    return terminal_result(
                        "error",
                        "最终回答连续两次未能使用本轮真实生成的图像。",
                        "solution_evidence_error",
                    )
                allowed = [item["url"] for item in artifacts if item.get("url")]
                repair_instruction = (
                    "The request explicitly requires a visual and no artifact exists. "
                    "Return one <execute> program now that computes the requested evidence, "
                    "saves it below SEVIR_ARTIFACT_DIR, and assigns a compact result."
                    if visual_required and not allowed
                    else (
                        "Return one corrected <solution> block, use only these exact "
                        "registered artifact URLs: "
                        + _compact(allowed, limit=2_000)
                        + ". Do not invent links or execute more code in this repair turn."
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The proposed solution failed evidence validation: "
                            + artifact_error
                            + ". Nothing in that solution was accepted. "
                            + repair_instruction
                        ),
                    }
                )
                continue
            # A multi-step analysis should reflect on its synthesis, not stop
            # at the first fluent draft.  The audit costs one full model call
            # (the most expensive token-generation step of the request), so it
            # is reserved for genuinely long analyses: with fewer than three
            # successful executions the freshest observation still carries
            # essentially all evidence and the draft can ship as is.  It is
            # also skipped when the remaining budget cannot cover the audit
            # plus a corrected answer; shipping the draft beats dying mid-audit.
            audit_deadline = request_deadline()
            enough_time_for_audit = (
                audit_deadline is None
                or (audit_deadline - time.monotonic()) >= REVIEW_TIME_FLOOR_SECONDS
            )
            if (
                successful_executions >= 3
                and not solution_reviewed
                and step < max(1, int(max_steps))
                and enough_time_for_audit
            ):
                solution_reviewed = True
                remaining_executions = max(
                    0, max(1, int(max_executions)) - execution_attempts
                )
                trace.append(
                    {
                        "attempt": step,
                        "action": "solution_review",
                        "status": "pending",
                        "elapsed_ms": model_elapsed_ms,
                        "summary": "复杂分析进入一次最终目标与证据自审",
                    }
                )
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": _action_message(content, action, body),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Treat that solution as a draft and perform one final reflective "
                                "audit against the exact user question and the returned result. "
                                "For every requested comparison, verify that the metric measures "
                                "the named concept for both groups rather than a nearby proxy; "
                                "verify the described sample cohort matches the actual selection; "
                                "and verify every number and figure URL exists in observations. "
                                f"The remaining code-execution budget is {remaining_executions}. "
                                + ("If evidence is missing, return one focused <execute> repair. "
                                   if remaining_executions > 0
                                   else "No further execution is allowed; do not return <execute>. ")
                                + "Otherwise return a corrected <solution> "
                                "that explicitly states any limitation. Do not merely repeat the draft."
                            ),
                        },
                    ]
                )
                continue
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

        budget_deadline = request_deadline()
        remaining_seconds = (
            budget_deadline - time.monotonic()
            if budget_deadline is not None
            else None
        )
        time_reserve = (
            LAST_FIGURE_RESERVE_SECONDS
            if visual_required and not artifacts
            else SOLUTION_TIME_RESERVE_SECONDS
        )
        if (
            remaining_seconds is not None
            and remaining_seconds < time_reserve
        ):
            trace.append({
                "attempt": step,
                "action": "time_budget",
                "status": "blocked",
                "elapsed_ms": model_elapsed_ms,
                "summary": "剩余时间优先保留给最终回答，停止新的代码执行",
            })
            messages.extend([
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        "The request time budget is almost exhausted, so no further code "
                        "execution is allowed. Using the observations and artifacts already "
                        "obtained, return one <solution> block now; briefly state any "
                        "limitation caused by the computation that did not finish."
                    ),
                },
            ])
            continue

        if tool_cost_blocked:
            trace.append({
                "attempt": step,
                "action": "tool_cost_budget",
                "status": "blocked",
                "elapsed_ms": model_elapsed_ms,
                "summary": "运行时工具成本预算不足，停止新的取证调用",
                "tool_cost_summary": latest_tool_cost_summary,
            })
            messages.extend([
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        "The runtime tool-cost budget does not permit another backend call. "
                        "Do not execute again. Use retained evidence and return one <solution>."
                    ),
                },
            ])
            continue

        if execution_attempts >= max(1, int(max_executions)):
            trace.append({
                "attempt": step,
                "action": "execution_budget",
                "status": "blocked",
                "elapsed_ms": model_elapsed_ms,
                "summary": "已达到代码执行安全上限，要求Agent依据已有观察完成回答",
            })
            messages.extend([
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        "The code-execution safety budget is exhausted. Do not execute again. "
                        "Use the observations already obtained and return one <solution> block."
                    ),
                },
            ])
            continue

        undocumented = sorted(
            _capabilities_used_in_code(body) - revealed_capabilities
        )
        if undocumented:
            required = undocumented[0]
            trace.append({
                "attempt": step,
                "action": "contract_required",
                "status": "blocked",
                "elapsed_ms": model_elapsed_ms,
                "summary": f"执行前需要先展开能力契约：{required}",
                "capabilities": undocumented,
            })
            messages.extend([
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        "That program was not executed because it uses a capability whose "
                        "public contract has not been revealed. Return exactly "
                        f"<expand>{required}</expand> now; do not guess its fields or signature."
                    ),
                },
            ])
            continue

        execution_started = time.perf_counter()
        execution_attempts += 1
        execution = executor.execute(
            body,
            context,
            execution_request=request,
        )
        # Syntax/safety preflight rejection is a model turn and an error, but
        # it is not a code execution: no Worker program ran and no compute
        # budget was consumed.  Counting it against ``max_executions`` denied
        # the reflective repair that the preflight error was designed to
        # enable.
        if (
            isinstance(execution.get("preflight"), dict)
            and execution["preflight"].get("execution_started") is False
        ):
            execution_attempts -= 1
        # A worker transport can be successful while the generated program
        # explicitly reports a domain failure in ``result``.  Promote that
        # value at the Agent boundary so it enters the normal repair/stop
        # path instead of being treated as a valid observation.
        domain_result = execution.get("result")
        if (
            execution.get("status") in {"ok", "partial"}
            and isinstance(domain_result, dict)
            and str(domain_result.get("status", "")).strip().lower()
            in {"error", "failed", "unavailable"}
        ):
            execution["status"] = "error"
            execution["error_type"] = "DomainResultError"
            execution["error"] = str(
                domain_result.get("message")
                or domain_result.get("error")
                or domain_result.get("status")
            )
            execution["error_phase"] = "generated_result"
        execution_elapsed_ms = round((time.perf_counter() - execution_started) * 1000, 2)
        call_summaries = [
            {
                "api": call.get("api"),
                "arguments": call.get("arguments", {}),
                "status": call.get("status"),
                "cache_hit": call.get("cache_hit", False),
                "cost_units": call.get("cost_units", 0.0),
                "cumulative_cost_units": call.get("cumulative_cost_units", 0.0),
                "remaining_cost_units": call.get("remaining_cost_units"),
                "error": call.get("error"),
                "return_type": call.get("return_type"),
                "result_summary": _tool_result_summary(call),
                "evidence": call.get("result"),
            }
            for call in execution.get("calls", [])
            if isinstance(call, dict)
        ]
        successful_calls = sum(call.get("status") == "ok" for call in call_summaries)
        failed_calls = sum(call.get("status") != "ok" for call in call_summaries)
        tool_call_summary = {
            "total": len(call_summaries),
            "successful": successful_calls,
            "failed": failed_calls,
        }
        latest_tool_cost_summary = dict(execution.get("tool_cost_summary") or {})
        evidence_this_execution = bool(
            execution.get("evidence_acquired", successful_calls > 0)
        )
        evidence_calls_this_execution = int(
            execution.get("evidence_tool_call_count", successful_calls) or 0
        )
        postprocessing_succeeded = bool(
            execution.get(
                "postprocessing_succeeded",
                execution.get("status") in {"ok", "partial"},
            )
        )
        if evidence_this_execution:
            evidence_execution_count += 1
            evidence_tool_call_count += evidence_calls_this_execution
        if postprocessing_succeeded:
            postprocessing_success_count += 1
        if execution.get("error_type") == "ToolCostBudgetExceeded":
            tool_cost_blocked = True
        if execution.get("status") == "ok" and failed_calls:
            execution["status"] = "partial"
            execution["observation_quality"] = "partial"
        observation = {
            "status": execution.get("status"),
            "result": execution.get("result"),
            "stdout": execution.get("stdout", ""),
            "available_names": execution.get("available_names", []),
            "namespace_state": execution.get("namespace_state", {}),
            "failed_value_shapes": execution.get("failed_value_shapes", {}),
            "calls": call_summaries,
            "tool_call_summary": tool_call_summary,
            "tool_cost_summary": latest_tool_cost_summary,
            "evidence_acquired": evidence_this_execution,
            "evidence_tool_call_count": evidence_calls_this_execution,
            "postprocessing_succeeded": postprocessing_succeeded,
            "generated_artifacts": execution.get("generated_artifacts", []),
            "error_type": execution.get("error_type"),
            "error": execution.get("error"),
            "traceback": execution.get("traceback"),
            "error_location": execution.get("error_location"),
            "failed_value_shapes": execution.get("failed_value_shapes", {}),
        }
        observations.append(observation)
        for artifact in _artifact_records(observation):
            if artifact["url"] not in {item["url"] for item in artifacts}:
                artifacts.append(artifact)

        if execution.get("status") in {"ok", "partial"}:
            successful_executions += 1
            consecutive_errors = 0
            summary = (
                "代码执行完成，但部分工具调用失败，需依据失败信息调整分析"
                if execution.get("status") == "partial"
                else "代码执行成功，已获得新的真实观测结果"
            )
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
                "tool_cost_summary": latest_tool_cost_summary,
                "evidence_acquired": evidence_this_execution,
                "postprocessing_succeeded": postprocessing_succeeded,
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
        # Zephyrus-style reflection depends on seeing the calculation result,
        # not merely knowing that code ran.  Values are already structurally
        # bounded by ``_prompt_value``; a 2.5k character prefix routinely cut
        # off the newest comparison statistics at the end of ``result`` and
        # forced the model to guess its conclusion, so the two newest
        # observations always stay verbatim.  Older observations are folded to
        # a short summary: each can carry up to 10k characters that every
        # later model call would otherwise re-read, even though the Worker
        # namespace still holds the real values for direct reuse.
        observation_positions = [
            index
            for index, item in enumerate(messages)
            if item.get("role") == "user"
            and str(item.get("content", "")).startswith("<observation>")
        ]
        for rank, position in enumerate(observation_positions[:-1]):
            messages[position]["content"] = _folded_observation(observations[rank])
        evidence_text = _compact(_observation_for_model(observation), limit=10_000)
        repair_capability_names = _capabilities_used_in_code(body)
        repair_capability_names.update(
            str(call.get("api") or "").split(".", 1)[0]
            for call in call_summaries
            if str(call.get("api") or "").split(".", 1)[0]
            in {"sevir", "geolocator", "cascast", "aurora"}
        )
        repair_capabilities = tuple(sorted(
            repair_capability_names if execution.get("status") == "error" else ()
        ))
        repair_contract = (
            "\n<repair-contracts names=\""
            + ",".join(repair_capabilities)
            + "\">\n"
            + compact_api_reference(repair_capabilities)
            + "\n</repair-contracts>\n"
            if repair_capabilities else ""
        )
        messages.extend(
            [
                {"role": "assistant", "content": _action_message(content, action, body)},
                {
                    "role": "user",
                    "content": (
                        "<observation>\n" + evidence_text + "\n</observation>\n"
                        + repair_contract
                        + "Continue the same analysis. Decide whether the user's goal is now "
                        "complete; return <solution> if it is, otherwise execute one genuinely "
                        "useful next analysis. Use the actual traceback when repairing an error "
                        "and do not repeat a successful call without new evidence. The Worker "
                        "namespace persists across turns, but failed executions are transactional: "
                        "namespace_state is authoritative: reuse only safe_to_reuse; names under "
                        "created_or_updated_names were committed by a successful transaction, while "
                        "uncommitted_names and not_reached_assignment_names must be recreated. Values "
                        "listed under failed_value_shapes expose exact diagnostic dimensions but were "
                        "rolled back. The result slot is per-attempt and is never inherited. "
                        "When repairing, return one small complete program that fixes the failed line. "
                        "Never replace retained real data "
                        "with simulated or random values."
                    ),
                },
            ]
        )
        _fold_capability_contracts(messages)
        expanded_capabilities.clear()

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
    execution_request: ExecutionRequest | None = None,
    conversation_context: str = "",
    executor: ExecutionClient | None = None,
    max_steps: int = 8,
    max_executions: int = 5,
    max_error_retries: int = 3,
) -> dict[str, Any]:
    """Run one Agent session and always terminate an internally owned Worker."""

    owned_executor = executor is None
    executor_timeout = float(os.getenv("SEVIR_AGENT_EXECUTION_TIMEOUT_SECONDS", "180"))
    active_executor = executor or ExecutionClient(timeout_seconds=executor_timeout, memory_limit_mb=4096)
    try:
        return _run_unified_agent_session(
            provider,
            model,
            question,
            context,
            complete,
            data_descs=data_descs,
            selected_tools=selected_tools,
            execution_request=execution_request,
            conversation_context=conversation_context,
            executor=active_executor,
            max_steps=max_steps,
            max_executions=max_executions,
            max_error_retries=max_error_retries,
        )
    finally:
        if owned_executor:
            active_executor.close()
