from __future__ import annotations

import argparse
import base64
import json
import re
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
from sevir_world.api.unified_service import request_budget_seconds
from sevir_world.agent.visual_contract import visual_artifact_required
try:
    from .reviewed_questions_42 import follow_up_chain, question_tuples
except ImportError:  # Direct ``python scripts/run_acceptance_42.py`` execution.
    from reviewed_questions_42 import follow_up_chain, question_tuples


ROOT = Path(__file__).resolve().parents[1]
CASE_CONFIGS = sorted((ROOT / "data" / "demo_cases").glob("*.json"))


def discover_cached_event_id() -> str:
    """Choose an available regression asset without embedding an event ID."""

    cache_root = ROOT / "data" / "live_cascast_cache"
    upload_root = ROOT / "data" / "examples" / "uploads"
    suffix = "_s0_ens10.npz"
    for cache_path in sorted(cache_root.glob(f"*{suffix}")):
        event_id = cache_path.name[: -len(suffix)]
        history = upload_root / event_id / "history"
        if len(list(history.glob("*.png"))) >= 13:
            return event_id
    raise RuntimeError("no cached CasCast event with 13 history frames is available")


EVENT_ID = discover_cached_event_id()
HISTORY = ROOT / "data" / "examples" / "uploads" / EVENT_ID / "history"
SINGLE = sorted(HISTORY.glob("*.png"), key=lambda path: path.name)[-1]


def discover_demo_case(event_id: str) -> dict[str, Any]:
    """Select an explicit regional case configuration without a code default."""

    candidates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in CASE_CONFIGS
    ]
    eligible = [
        item for item in candidates
        if "regional_domain" in item and "time_contract" in item
    ]
    exact = [
        item for item in eligible
        if str(item.get("case_id") or "").upper() == str(event_id).upper()
    ]
    if len(exact) == 1:
        return exact[0]
    if len(eligible) == 1:
        return eligible[0]
    raise RuntimeError(
        "regional acceptance requires one unambiguous data/demo_cases configuration"
    )


DEMO_CASE = discover_demo_case(EVENT_ID)
AURORA_INITIALIZATION_TIME = str(
    DEMO_CASE["time_contract"]["aurora_initialization_time_utc"]
)
REGIONAL_DOMAIN = dict(DEMO_CASE["regional_domain"])

QUESTIONS = question_tuples()
FOLLOW_UPS = follow_up_chain()
FAILURE_TERMINATIONS = {
    "consecutive_execution_errors",
    "max_steps_without_solution",
    "model_api_terminal_failure",
}
STOP_ON_DEADLINE = threading.Event()
ACCEPTANCE_TIMEOUT_SECONDS = int(request_budget_seconds())
TRANSPORT_COMPLETION_GRACE_SECONDS = 5
REQUEST_PACING_SECONDS = 3.0
_REQUEST_PACING_LOCK = threading.Lock()
_LAST_REQUEST_STARTED = 0.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def frame_payload(path: Path) -> dict[str, Any]:
    """Convert one cached display PNG to the upload API's raw VIL contract."""
    with Image.open(path) as image:
        grayscale = image.convert("L")
        if grayscale.size != (384, 384):
            raise ValueError(f"{path.name} is {grayscale.size}, expected 384x384")
        raw = grayscale.tobytes()
    return {
        "filename": path.name,
        "width": 384,
        "height": 384,
        "pixels_base64": base64.b64encode(raw).decode("ascii"),
        "grayscale_source": True,
    }


def post(base_url: str, route: str, payload: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + route,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=max(1, int(timeout))) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("API response is not an object")
    return value


def _compact_trace(body: dict[str, Any]) -> dict[str, Any]:
    agent_steps = []
    for item in body.get("agent_trace", []) or []:
        if not isinstance(item, dict):
            continue
        agent_steps.append({
            "attempt": item.get("attempt"),
            "action": item.get("action"),
            "status": item.get("status"),
            "error_type": item.get("error_type"),
            "error": str(item.get("error", ""))[:1000],
            "elapsed_ms": item.get("elapsed_ms"),
            "summary": item.get("summary"),
        })
    tools = []
    for item in body.get("tool_runs", []) or []:
        if not isinstance(item, dict):
            continue
        tools.append({
            "api": item.get("api"),
            "status": item.get("status"),
            "cache_hit": bool(item.get("cache_hit", False)),
            "error": str(item.get("error", ""))[:1000],
            "arguments": item.get("arguments", {}),
        })
    return {
        "agent_trace": agent_steps,
        "tool_runs": tools,
        "llm_timings": body.get("llm_timings", []),
        "request_trace": body.get("request_trace"),
        "technical_details": body.get("technical_details"),
        "diagnostic": body.get("diagnostic"),
    }


def _result_status(
    body: dict[str, Any], *, visual_required: bool = False
) -> tuple[str, str]:
    termination = str(body.get("termination_reason", ""))
    route = str(body.get("route_mode", ""))
    answer = str(body.get("answer", "")).strip()
    technical_text = json.dumps(
        body.get("technical_details") or {}, ensure_ascii=False, default=str
    ).lower()
    if "allocationquota.freetieronly" in technical_text or "free quota exhausted" in technical_text:
        return "blocked", "model_free_quota_exhausted"
    if termination == "model_api_arrearage" or "arrearage" in technical_text or "overdue-payment" in technical_text:
        return "blocked", "model_account_arrearage"
    if termination == "model_api_terminal_failure":
        return "blocked", "model_api_terminal_failure"
    if termination in FAILURE_TERMINATIONS:
        return "failed", "agent_or_protocol_failure"
    technical = body.get("technical_details") or {}
    agent_error = technical.get("agent_error") if isinstance(technical, dict) else None
    if agent_error:
        message = str(agent_error.get("message", "")) if isinstance(agent_error, dict) else str(agent_error)
        if "Arrearage" in message or "overdue-payment" in message:
            return "blocked", "model_account_arrearage"
        return "failed", "model_or_agent_exception"
    if body.get("status") == "ok":
        # ``ok`` is only the Agent transport state.  Acceptance requires a
        # user-facing synthesis and at least one inspectable visual artifact;
        # raw protocol/code or an empty artifact list is not a pass.
        if re.search(r"<execute>|```python|^import\s|^from\s", answer, re.IGNORECASE | re.MULTILINE):
            return "failed", "raw_protocol_or_code_in_answer"
        if (
            visual_required
            and not body.get("artifacts")
            and not body.get("artifact_count")
        ):
            return "failed", "missing_visual_artifact"
        return "passed", ""
    if body.get("status") == "partial":
        return "partial", "partial_result"
    return "failed", str(body.get("status") or "unknown_status")


def _payload(
    number: int,
    kind: str,
    question: str,
    ids: dict[str, str],
    session_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_source": "catalog",
        "data_policy": "cached",
        "upload_id": "",
        "event_id": "",
        "request_id": uuid.uuid4().hex,
        "session_id": session_id,
        "trace": True,
        "question": f"【42题验收Q{number:02d}】{question}",
    }
    if 33 <= number <= 42:
        payload["study_case_id"] = session_id
        payload["study_stage"] = f"Q{number:02d}"
        payload["visual_artifact_required"] = True
        payload["visual_artifact_required"] = True
    if kind == "single":
        payload.update({"data_source": "upload", "upload_id": ids["single"]})
    elif kind in {"sequence", "invalid_sequence"}:
        if kind == "invalid_sequence":
            payload.update({
                "data_source": "upload_sequence",
                "upload_id": ids.get("invalid_sequence", ""),
            })
        else:
            # Bind sequence tests to the durable event cache. Registering the
            # same pixels again creates a new upload id and would otherwise
            # bypass the already-computed event-level CasCast ensemble.
            payload.update({
                "data_source": "catalog",
                "event_id": EVENT_ID,
            })
    elif kind == "event_environment":
        payload.update({
            "data_source": "environment",
            "data_policy": "cached",
            "event_id": "",
            "history_before_hours": 6,
            "history_after_hours": 24,
            "forecast_leads_hours": [6, 12, 18, 24],
        })
        if number in {27, 33}:
            # Only the first stage establishes the physical regional request.
            # Later stages consume and refine the persisted evidence ledger.
            payload.update({
                "requested_time_utc": AURORA_INITIALIZATION_TIME,
                "aurora_initialization_time_utc": AURORA_INITIALIZATION_TIME,
                "aurora_region": REGIONAL_DOMAIN,
            })
    elif kind == "sequence_environment":
        payload.update({
            "data_policy": "cached",
            "data_source": "catalog",
            "history_before_hours": 24,
            "history_after_hours": 24,
            "forecast_leads_hours": [6, 12, 18, 24],
        })
    elif kind == "live_place":
        payload.update({
            "data_policy": "live",
            "requested_time_utc": "2019-08-01T21:00:00Z",
            "latitude": 34.7659,
            "longitude": -84.7699,
        })
    return payload


def run_question(
    base_url: str,
    number: int,
    kind: str,
    question: str,
    ids: dict[str, str],
    session_id: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    global _LAST_REQUEST_STARTED
    if STOP_ON_DEADLINE.is_set():
        return {
            "number": number,
            "question": question,
            "kind": kind,
            "status": "not_started",
            "failure_class": "stopped_after_deadline",
            "elapsed_seconds": 0,
            "diagnosis_required": True,
        }
    # Batch evaluation is not a load test. Space fixed-provider starts so
    # rate-limit/connection long tails do not get misclassified as Agent or
    # tool failures. This wait occurs before the measured per-request budget.
    with _REQUEST_PACING_LOCK:
        wait_seconds = REQUEST_PACING_SECONDS - (time.monotonic() - _LAST_REQUEST_STARTED)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_REQUEST_STARTED = time.monotonic()
    payload = _payload(number, kind, question, ids, session_id)
    request_id = str(payload["request_id"])
    started = time.perf_counter()
    try:
        # The Agent itself has a configured budget after admission.  Parallel
        # questions may wait in the server's admission queue, so the HTTP
        # transport gets extra time without changing the Agent budget.
        # Keep transport wait aligned with the per-question acceptance budget.
        # Agent work is bounded by the shared acceptance deadline.  The HTTP
        # read gets a small completion-only grace window so the server can
        # return its structured deadline trace instead of losing diagnostics
        # in a client/server timeout race at exactly the same instant.
        body = post(
            base_url,
            "/api/run",
            payload,
            timeout=timeout + TRANSPORT_COMPLETION_GRACE_SECONDS,
        )
        # Every Study Case stage must leave an inspectable evidence figure.
        # This matches the request fact sent to the Agent in _payload, so the
        # generation contract and this acceptance gate cannot disagree.
        visual_required = (33 <= number <= 42) or visual_artifact_required(question)
        status, failure_class = _result_status(
            body, visual_required=visual_required
        )
        elapsed = time.perf_counter() - started
        result = {
            "number": number,
            "question": question,
            "kind": kind,
            "status": status,
            "failure_class": failure_class,
            "elapsed_seconds": round(elapsed, 2),
            "server_elapsed_ms": body.get("elapsed_ms"),
            "route_mode": body.get("route_mode"),
            "termination_reason": body.get("termination_reason"),
            "answer": body.get("answer", ""),
            "artifacts": body.get("artifacts", []),
            "artifact_count": int(body.get("artifact_count") or len(body.get("artifacts", []) or [])),
            "visual_artifact_required": visual_required,
            "event_resolution": body.get("event_resolution"),
            "capability_request": body.get("capability_request"),
            "evidence_bundle": body.get("evidence_bundle"),
            "request_id": body.get("request_id"),
            "trace_file": body.get("trace_file"),
            "session_id": body.get("session_id", session_id),
            **_compact_trace(body),
        }
        request_budget = body.get("request_budget") or {}
        agent_deadline_reached = bool(request_budget.get("deadline_reached"))
        if agent_deadline_reached:
            STOP_ON_DEADLINE.set()
            result.update({
                "status": "deadline_exceeded",
                "failure_class": "request_exceeded_90s_budget",
                "diagnosis_required": True,
            })
        return result
    except Exception as error:
        reason = error.reason if isinstance(error, URLError) else None
        timed_out = isinstance(error, (TimeoutError, socket.timeout)) or isinstance(
            reason, (TimeoutError, socket.timeout)
        )
        if timed_out:
            STOP_ON_DEADLINE.set()
        return {
            "number": number,
            "question": question,
            "kind": kind,
            "status": "deadline_exceeded" if timed_out else "request_error",
            "failure_class": (
                "request_exceeded_90s_budget" if timed_out else "request_exception"
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "request_id": request_id,
            "error": f"{type(error).__name__}: {error}",
            "diagnosis_required": timed_out,
        }


def emit(result: dict[str, Any]) -> None:
    number = result.get("number")
    status = result.get("status") or "unknown"
    elapsed = result.get("elapsed_seconds")
    termination = result.get("termination_reason") or "—"
    print(
        f"Q{number} | {status} | {elapsed if elapsed is not None else '—'} 秒 | 结束原因：{termination}",
        flush=True,
    )


def run_parallel(
    base_url: str,
    numbers: list[int],
    ids: dict[str, str],
    workers: int,
    timeout: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    worker_count = max(1, workers)
    pending_numbers = iter(numbers)
    pool = ThreadPoolExecutor(max_workers=worker_count)
    futures: dict[Any, int] = {}

    def submit_next() -> bool:
        if STOP_ON_DEADLINE.is_set():
            return False
        try:
            number = next(pending_numbers)
        except StopIteration:
            return False
        kind, question = QUESTIONS[number]
        future = pool.submit(
            run_question,
            base_url,
            number,
            kind,
            question,
            ids,
            f"acceptance42-q{number:02d}-{uuid.uuid4()}",
            timeout=timeout,
        )
        futures[future] = number
        return True

    try:
        for _ in range(min(worker_count, len(numbers))):
            submit_next()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                futures.pop(future, None)
                result = future.result()
                results.append(result)
                emit(result)
                if STOP_ON_DEADLINE.is_set():
                    for queued in futures:
                        queued.cancel()
                    return results
                submit_next()
    finally:
        pool.shutdown(wait=not STOP_ON_DEADLINE.is_set(), cancel_futures=True)
    return results


def write_manifest(
    output: Path | None,
    ids: dict[str, str],
    results: list[dict[str, Any]],
    *,
    started_at: str,
    stopped_early: bool,
) -> None:
    results.sort(key=lambda item: int(item["number"]))
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "asset_event_id": EVENT_ID if ids else None,
        "study_case_event_selection": "agent_selected_from_evidence",
        "question_count": 42,
        "result_count": len(results),
        "status_counts": counts,
        "stopped_early": stopped_early,
        "stop_reason": "request_exceeded_90s_budget" if stopped_early else "",
        "diagnosis_required": stopped_early,
        "ids": ids,
        "results": results,
    }
    if output is None:
        print(
            "SUMMARY " + json.dumps({
                key: manifest[key] for key in (
                    "question_count", "result_count", "status_counts",
                    "stopped_early", "stop_reason", "diagnosis_required",
                )
            }, ensure_ascii=False),
            flush=True,
        )
        return
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def stop_after_deadline(
    output: Path | None,
    ids: dict[str, str],
    results: list[dict[str, Any]],
    *,
    started_at: str,
) -> bool:
    if not STOP_ON_DEADLINE.is_set():
        return False
    write_manifest(
        output, ids, results, started_at=started_at, stopped_early=True
    )
    print(
        "STOP：已有请求超过 90 秒预算；未开始的问题已取消，请先查看 manifest.json 诊断。",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8024")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-write", action="store_true",
        help="执行完整验收并输出逐题结果，但不写结果文件",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="HTTP验收并发数；固定模型接口默认串行，确认提供方支持并发后方可提高",
    )
    parser.add_argument(
        "--only", default="",
        help="仅运行逗号分隔的题号；Study Case 的33—42仍按题号串行",
    )
    parser.add_argument(
        "--timeout", type=int, default=ACCEPTANCE_TIMEOUT_SECONDS,
        help="单题硬超时；缓存验收不得高于90秒",
    )
    args = parser.parse_args()
    STOP_ON_DEADLINE.clear()
    only_numbers = sorted({
        int(value.strip()) for value in str(args.only).split(",") if value.strip()
    })
    invalid = [number for number in only_numbers if number not in QUESTIONS or number == 30]
    if invalid:
        raise ValueError(f"--only contains unsupported question numbers: {invalid}")
    study_only = bool(only_numbers) and all(33 <= number <= 42 for number in only_numbers)
    started_at = datetime.now().isoformat()
    args.timeout = min(ACCEPTANCE_TIMEOUT_SECONDS, max(1, args.timeout))
    output: Path | None = None if args.no_write else args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = {path.name for path in output.iterdir()}
            if existing - {"uploads.json", "manifest.json"}:
                raise FileExistsError(f"acceptance output is not empty: {output}")
        else:
            output.mkdir(parents=True, exist_ok=False)

    ids: dict[str, str] = {}
    if not study_only:
        history = sorted(HISTORY.glob("*.png"), key=lambda path: path.name)
        if len(history) != 13:
            raise RuntimeError(f"expected 13 history frames, got {len(history)}")
        single = post(args.base_url, "/api/upload", {
            **frame_payload(SINGLE),
            "event_id": EVENT_ID,
        })
        sequence = post(args.base_url, "/api/upload-sequence", {
            "frames": [frame_payload(path) for path in history],
            "event_id": EVENT_ID,
        })
        invalid_sequence = post(args.base_url, "/api/upload-sequence", {
            "frames": [frame_payload(path) for path in history[:12]],
            "event_id": EVENT_ID,
        })
        ids = {
            "single": str(single.get("upload_id", "")),
            "sequence": str(sequence.get("upload_id", "")),
            "invalid_sequence": str(invalid_sequence.get("upload_id", "")),
        }
    if output is not None and not study_only:
        (output / "uploads.json").write_text(
            json.dumps({
                "event_id": EVENT_ID,
                "single": single,
                "sequence": sequence,
                "invalid_sequence": invalid_sequence,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    results: list[dict[str, Any]] = []
    if only_numbers:
        if all(33 <= number <= 42 for number in only_numbers):
            session_id = f"acceptance42-foundation-{uuid.uuid4()}"
            for number in only_numbers:
                result = run_question(
                    args.base_url, number, *QUESTIONS[number], ids,
                    session_id, timeout=args.timeout,
                )
                results.append(result)
                emit(result)
                if stop_after_deadline(output, ids, results, started_at=started_at):
                    return
        else:
            results.extend(run_parallel(
                args.base_url, only_numbers, ids,
                min(args.workers, len(only_numbers)), args.timeout,
            ))
        if stop_after_deadline(output, ids, results, started_at=started_at):
            return
        write_manifest(
            output, ids, results, started_at=started_at, stopped_early=False
        )
        print(f"DONE {output or 'no-write'}", flush=True)
        return

    # Only questions without dependency edges run together.  Q29 depends on
    # the ensemble consensus from Q15, so it must not be launched in this
    # initial batch merely because both are cached analyses.
    results.extend(run_parallel(
        args.base_url,
        [1, 2, 3, 4, 5, 6, 25, 26, 27, 28, 32],
        ids,
        args.workers,
        args.timeout,
    ))
    if stop_after_deadline(output, ids, results, started_at=started_at):
        return

    if ids["invalid_sequence"]:
        result = run_question(
            args.base_url, 31, *QUESTIONS[31], ids,
            f"acceptance42-q31-{uuid.uuid4()}", timeout=args.timeout,
        )
    else:
        result = {
            "number": 31,
            "question": QUESTIONS[31][1],
            "kind": "invalid_sequence",
            "status": "passed" if invalid_sequence.get("status") in {"error", "invalid", "unavailable"} else "failed",
            "failure_class": "",
            "elapsed_seconds": 0,
            "termination_reason": "upload_validation",
            "technical_details": invalid_sequence,
        }
    results.append(result)
    emit(result)
    if stop_after_deadline(output, ids, results, started_at=started_at):
        return

    # Observation-only questions can run together.
    results.extend(run_parallel(args.base_url, [7, 8], ids, min(2, args.workers), args.timeout))
    if stop_after_deadline(output, ids, results, started_at=started_at):
        return

    # Warm/reuse CasCast once for the newly registered sequence, then parallelize
    # the remaining read/analysis questions against the cached result.
    q9 = run_question(
        args.base_url, 9, *QUESTIONS[9], ids,
        f"acceptance42-q09-{uuid.uuid4()}", timeout=args.timeout,
    )
    results.append(q9)
    emit(q9)
    if stop_after_deadline(output, ids, results, started_at=started_at):
        return
    # Preserve the dependency graph in three frontiers.  Items inside one
    # frontier are independent and may run in parallel; later frontiers consume
    # evidence or conversational context established by the earlier ones.
    for frontier in (
        [10, 11, 12, 13, 14, 15, 21, 23],
        [16, 19, 20, 22, 24],
        [17, 18],
    ):
        results.extend(run_parallel(
            args.base_url, frontier, ids, min(3, args.workers), args.timeout
        ))
        if stop_after_deadline(output, ids, results, started_at=started_at):
            return

    # Q30 is one context-retention case containing four serial follow-ups.
    follow_session = f"acceptance42-q30-{uuid.uuid4()}"
    seed = run_question(
        args.base_url, 29, *QUESTIONS[29], ids, follow_session, timeout=args.timeout
    )
    if STOP_ON_DEADLINE.is_set():
        results.append(seed)
        emit(seed)
        stop_after_deadline(output, ids, results, started_at=started_at)
        return
    follow_results = []
    for index, follow_question in enumerate(FOLLOW_UPS, start=1):
        turn = run_question(
            args.base_url,
            30,
            "sequence",
            follow_question,
            ids,
            follow_session,
            timeout=args.timeout,
        )
        turn["subquestion"] = f"30.{index}"
        follow_results.append(turn)
        emit({**turn, "number": f"30.{index}"})
        if STOP_ON_DEADLINE.is_set():
            results.append({
                "number": 30,
                "question": "连续修改阈值、历史条件、县域范围和生活建议时间。",
                "kind": "follow_up_chain",
                "status": "deadline_exceeded",
                "failure_class": "request_exceeded_90s_budget",
                "elapsed_seconds": round(
                    sum(item["elapsed_seconds"] for item in follow_results), 2
                ),
                "seed": seed,
                "follow_ups": follow_results,
                "session_id": follow_session,
                "diagnosis_required": True,
            })
            stop_after_deadline(output, ids, results, started_at=started_at)
            return
    follow_status = (
        "passed" if all(item["status"] == "passed" for item in follow_results)
        else "blocked" if all(item["status"] == "blocked" for item in follow_results)
        else "failed"
    )
    follow_failure_classes = {
        str(item.get("failure_class", "")) for item in follow_results
        if item.get("failure_class")
    }
    results.append({
        "number": 30,
        "question": "连续修改阈值、历史条件、县域范围和生活建议时间。",
        "kind": "follow_up_chain",
        "status": follow_status,
        "failure_class": (
            "" if follow_status == "passed"
            else (
                next(iter(follow_failure_classes))
                if follow_status == "blocked" and len(follow_failure_classes) == 1
                else "model_api_blocked"
            ) if follow_status == "blocked"
            else "follow_up_turn_failed"
        ),
        "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in follow_results), 2),
        "seed": seed,
        "follow_ups": follow_results,
        "session_id": follow_session,
    })

    # Preserve the teacher-requested region-discovery -> local-tracking
    # chronology and shared context for the configured regional study case.
    research_session = f"acceptance42-foundation-{uuid.uuid4()}"
    for number in range(33, 43):
        result = run_question(
            args.base_url, number, *QUESTIONS[number], ids,
            research_session, timeout=args.timeout,
        )
        results.append(result)
        emit(result)
        if stop_after_deadline(output, ids, results, started_at=started_at):
            return

    write_manifest(
        output, ids, results, started_at=started_at, stopped_early=False
    )
    print(f"DONE {output or 'no-write'}", flush=True)


if __name__ == "__main__":
    main()
