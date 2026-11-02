from __future__ import annotations

"""Transport-independent boundary for the single adaptive Agent path."""

from dataclasses import dataclass, replace
import time
import uuid
from typing import Any, Callable, Mapping

from sevir_world.agent.contracts import ExecutionRequest, ExecutionRuntime
from sevir_world.agent.context import conversation_context
from sevir_world.agent.session import load_session, normalize_session_id
from sevir_world.agent.study import rebuild_study_case
from sevir_world.agent.unified import AgentCancelled
from sevir_world.request.task_runner import TaskOutcome
from sevir_world.request.response_pipeline import finalize_agent_response
from sevir_world.agent.service import execute_unified_request
from sevir_world.request.assets import bind_request_assets


@dataclass(frozen=True)
class ExecutionDependencies:
    providers: Mapping[str, dict[str, str]]
    select_model: Callable[[Mapping[str, dict[str, str]]], tuple[str, str]]
    complete: Callable[
        [dict[str, str], str, list[dict[str, str]], float], dict[str, Any]
    ]
    summarize: Callable | None = None


def prepare_question_payload(
    payload: dict[str, Any],
    *,
    runtime: ExecutionRuntime | None = None,
) -> tuple[dict[str, Any], bool, ExecutionRequest]:
    """Resolve request facts only; do not classify or compile an execution path."""

    active_runtime = runtime or ExecutionRuntime()
    session_id = normalize_session_id(payload.get("session_id"))
    payload["session_id"] = session_id
    session_state = load_session(session_id)
    session_state["study_case"] = rebuild_study_case(
        session_id, session_state.get("turns", []), case_id=payload.get("study_case_id"),
    )
    if active_runtime.is_cancelled():
        raise AgentCancelled("client disconnected during asset binding")

    payload.setdefault("generate_visual", False)

    continuation = payload.get("continue_study_case", False)
    if not isinstance(continuation, bool):
        raise ValueError("continue_study_case must be a boolean")
    study_case = session_state["study_case"]
    requested_case = payload.get("study_case_id")
    case_has_turns = any(
        (turn.get("study_case_id") or session_id) == study_case["case_id"]
        for turn in session_state.get("turns", [])
    )
    inherit_study = case_has_turns and (
        requested_case == study_case["case_id"] if requested_case else continuation
    )
    if inherit_study:
        payload.setdefault("study_case_id", study_case["case_id"])
        active_upload = session_state.get("active_upload", {})
        has_upload = bool(payload.get("upload_id") or payload.get("sequence_upload_id"))
        if not has_upload:
            for key in ("upload_id", "sequence_upload_id", "data_source"):
                if active_upload.get(key):
                    payload[key] = active_upload[key]

    request = bind_request_assets(
        payload,
        selected_subject=study_case["decisions"].get("selected_subject") if inherit_study else None,
    )
    return session_state, inherit_study, request


def execute_request(
    *,
    question: str,
    payload: dict[str, Any],
    runtime: ExecutionRuntime,
    dependencies: ExecutionDependencies,
) -> TaskOutcome:
    """Bind facts, run the one Agent loop, and finalize exactly once."""

    request_id = runtime.request_id
    trace = runtime.trace
    started_perf_counter = runtime.started_perf_counter

    trace.append_live_step({
        "attempt": 0, "action": "request_received", "status": "ok",
        "summary": "服务端已接收问题并创建本轮请求。", "elapsed_ms": 0,
    }, progress=0.04)
    trace.update_activity(
        stage="asset_binding", summary="正在校验结构化请求事实与上传资料。",
        progress=0.08, action="asset_binding",
    )
    session_state, inherited, request = prepare_question_payload(payload, runtime=runtime)
    session_id = str(payload["session_id"])
    trace.append_live_step({
        "attempt": 0, "action": "asset_binding", "status": "ok",
        "summary": "请求事实已绑定，进入统一 Agent。",
        "elapsed_ms": round((time.perf_counter() - started_perf_counter) * 1000, 2),
    }, progress=0.12)

    if not inherited and not str(payload.get("study_case_id") or "").strip():
        token = uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{request_id}").hex[:16]
        payload["study_case_id"] = f"{session_id}-{token}"

    provider_name, model = dependencies.select_model(dependencies.providers)
    payload["provider"] = provider_name
    payload["model"] = model
    trace.add(current_stage="unified_agent", progress=0.15)
    analysis_started = time.perf_counter()
    raw = execute_unified_request(
        provider_name, model, question,
        providers=dependencies.providers, complete=dependencies.complete,
        runtime=runtime,
        execution_request=replace(
            request,
            evidence_state=session_state["study_case"] if inherited else {},
            conversation_context=conversation_context(session_state, payload["study_case_id"]) if inherited
            else {"case_id": payload["study_case_id"], "turns": []},
        ),
        summarize=dependencies.summarize,
    )
    presentation = {
        "event_id": str(payload.get("event_id", "")),
        "generate_visual": payload["generate_visual"],
    }
    result, turn_commit = finalize_agent_response(
        raw, payload=payload, question=question, request_id=request_id,
        provider_name=provider_name, model_name=model,
        started=started_perf_counter, analysis_started=analysis_started,
        trace=trace, presentation=presentation,
    )
    return replace(TaskOutcome.from_value(result), turn_commit=turn_commit)


