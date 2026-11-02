from __future__ import annotations

"""Thin HTTP boundary for the single unified Agent."""

import hashlib
import json
import os
import re
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


from weather_agent.agent.session import (
    available_session_ids,
    latest_session_id,
    load_session,
    normalize_session_id,
    normalize_turn_id,
)
from weather_agent.agent.case_memory import CaseMemoryStore, case_memory_path
from weather_agent.agent.session import delete_session_turn, merge_sessions
from weather_agent.model_client import (
    FIXED_MODEL_NAME,
    FIXED_PROVIDER_NAME,
    PROVIDERS, fixed_model_selection, post_chat_completion, post_context_summary, provider_status,
)
from weather_agent.request.observability import (
    metrics_snapshot,
    request_snapshot,
)
from weather_agent.request.service import ExecutionDependencies
from weather_agent.request.task_requests import submission_arguments, retry_arguments
from weather_agent.api.status import runtime_status
from weather_agent.request.task_runner import (
    TASK_QUEUE_API_SCHEMA_VERSION,
    TaskExecutor,
    TaskRunner,
)
from weather_agent.request.agent_task_executor import UnifiedAgentTaskExecutor
from weather_agent.persistence.task_store import (
    IdempotencyConflict,
    IllegalTaskTransition,
    RetryNotAllowed,
    TaskNotFound,
    TaskStore,
    TaskStoreError,
)
from weather_agent.agent.service import (
    request_budget_seconds,
)
from weather_agent.agent.study import evidence_index, rebuild_study_case
from weather_agent.cache.registry import CacheRegistry
from weather_agent.request.uploads import (
    list_uploaded_records,
    register_uploaded_vil,
    register_uploaded_vil_sequence,
)
from weather_agent.tools.visualization.upload_frames import (
    uploaded_frame,
    uploaded_sequence_frame,
)
from weather_agent.tools.cascast.thresholds import (
    DEFAULT_CASCAST_VIL_THRESHOLD,
    cascast_vil_threshold_payload,
)
from weather_agent.tools.visualization.renderer import (
    artifact_read_roots,
    render_png,
    session_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"


def task_links(request_id: str) -> dict:
    base = f"/api/tasks/{request_id}"
    return {"self": base, "events": f"{base}/events",
            "cancel": f"{base}/cancel", "retry": f"{base}/retry"}


class WeatherAgentRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {format % args}")

    def end_headers(self) -> None:
        # Serve current interface assets after each restart. Reusing an old app.js
        # can silently send an obsolete request contract (for example cached
        # instead of cache-first live), so application assets must always be
        # revalidated. Artifact PNGs keep their explicit cache policy.
        path = urlparse(self.path).path
        # Permit the static file:// interface to read the durable session and
        # artifact endpoints from the local Python service.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Idempotency-Key"
        )
        if path in {"/", "/index.html", "/app.js", "/styles.css"}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def send_json(self, payload: dict[str, Any], status: int = 200) -> bool:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False

    def send_png(self, body: bytes, *, immutable: bool = False) -> None:
        etag = f'"{hashlib.sha256(body).hexdigest()}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "public, max-age=31536000, immutable"
            if immutable else "private, max-age=300",
        )
        self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 5_000_000:
            raise ValueError("Request too large. Processed uploads must not exceed 5 MB.")
        value = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if not isinstance(value, dict):
            raise ValueError("The request body must be a JSON object.")
        return value

    def _task_runner(self) -> TaskRunner | None:
        runner = getattr(self.server, "task_runner", None)
        return runner if isinstance(runner, TaskRunner) else None

    def _send_task_error(self, error: BaseException) -> None:
        if isinstance(error, TaskNotFound):
            status = 404
            code = "task_not_found"
        elif isinstance(
            error,
            (
                IdempotencyConflict,
                IllegalTaskTransition,
                RetryNotAllowed,
            ),
        ):
            status = 409
            code = "task_conflict"
        elif isinstance(error, ValueError):
            status = 400
            code = "invalid_task_request"
        else:
            status = 503
            code = "task_store_unavailable"
        self.send_json({
            "schema_version": TASK_QUEUE_API_SCHEMA_VERSION,
            "error": str(error),
            "error_code": code,
        }, status)

    def _handle_task_get(self, parsed: Any) -> bool:
        path = str(parsed.path)
        events_match = re.fullmatch(
            r"/api/tasks/([0-9A-Za-z_-]{8,128})/events", path
        )
        task_match = re.fullmatch(
            r"/api/tasks/([0-9A-Za-z_-]{8,128})", path
        )
        session_match = re.fullmatch(
            r"/api/sessions/([0-9A-Za-z_-]{1,80})/active-tasks", path
        )
        if not (events_match or task_match or session_match):
            return False
        runner = self._task_runner()
        if runner is None:
            self.send_json({
                "schema_version": TASK_QUEUE_API_SCHEMA_VERSION,
                "error": "durable task service is not configured",
                "error_code": "task_service_unavailable",
            }, 503)
            return True
        try:
            if events_match:
                query = parse_qs(parsed.query)
                after_sequence = int(
                    str((query.get("after_sequence") or ["0"])[0])
                )
                response = runner.get_events(
                    events_match.group(1), after_sequence=after_sequence
                )
                response["links"] = task_links(events_match.group(1))
                self.send_json(response)
                return True
            if task_match:
                response = runner.get_task(task_match.group(1))
                response["links"] = task_links(task_match.group(1))
                self.send_json(response)
                return True
            assert session_match is not None
            session_id = normalize_session_id(session_match.group(1))
            tasks = runner.store.list_tasks(
                session_id=session_id,
                states=frozenset({
                    "queued",
                    "running",
                    "cancelling",
                }),
            )
            self.send_json({
                "schema_version": TASK_QUEUE_API_SCHEMA_VERSION,
                "session_id": session_id,
                "active_tasks": tasks,
                "active_task_count": len(tasks),
            })
        except (TaskStoreError, ValueError) as error:
            self._send_task_error(error)
        return True

    def _handle_task_post(self, path: str, payload: dict[str, Any]) -> bool:
        action_match = re.fullmatch(
            r"/api/tasks/([0-9A-Za-z_-]{8,128})/(cancel|retry)",
            path,
        )
        if path != "/api/tasks" and action_match is None:
            return False
        runner = self._task_runner()
        if runner is None:
            self.send_json({
                "schema_version": TASK_QUEUE_API_SCHEMA_VERSION,
                "error": "durable task service is not configured",
                "error_code": "task_service_unavailable",
            }, 503)
            return True
        try:
            if path == "/api/tasks":
                if not bool(
                    getattr(self.server, "task_execution_available", False)
                ):
                    self.send_json({
                        "schema_version": TASK_QUEUE_API_SCHEMA_VERSION,
                        "error": "asynchronous execution adapter is not configured",
                        "error_code": "task_execution_adapter_unavailable",
                        "execution_adapter": "not_configured",
                    }, 503)
                    return True
                arguments = submission_arguments(payload, self.headers.get("Idempotency-Key", ""))
                request_id = arguments["request_id"]
                response = runner.submit_task(**arguments)
                response["execution_adapter"] = "injected"
                response["links"] = task_links(request_id)
                self.send_json(response, 202)
                return True
            assert action_match is not None
            request_id, action = action_match.groups()
            if action == "cancel":
                response = runner.cancel_task(
                    request_id,
                    requested_by=str(payload.get("requested_by") or "user")[:128],
                )
                response["links"] = task_links(request_id)
                self.send_json(
                    response,
                    200 if response["task"].get("terminal") else 202,
                )
                return True
            arguments = retry_arguments(request_id, payload, self.headers.get("Idempotency-Key", ""))
            response = runner.retry_task(request_id, **arguments)
            response["links"] = task_links(arguments["request_id"])
            self.send_json(response, 202)
        except (TaskStoreError, ValueError) as error:
            self._send_task_error(error)
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        knowledge_match = re.fullmatch(r"/api/knowledge/chunks/([0-9]+)(/source)?", parsed.path)
        if knowledge_match:
            from ..tools.knowledge.source_view import source_response
            try:
                content_type, body = source_response(int(knowledge_match[1]), original=bool(knowledge_match[2]))
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)
            except (KeyError, ValueError, OSError, RuntimeError):
                self.send_json({"error": "knowledge source not available"}, 404)
            return
        if self._handle_task_get(parsed):
            return
        if parsed.path == "/api/metrics":
            self.send_json(metrics_snapshot())
            return
        request_match = re.fullmatch(
            r"/api/requests/([0-9A-Za-z_-]{8,128})", parsed.path
        )
        if request_match:
            request_id = request_match.group(1)
            try:
                runner = self._task_runner()
                if runner is None:
                    raise TaskNotFound(request_id)
                task = runner.store.get_task(request_id)
            except TaskNotFound:
                self.send_json({"error": "request not found"}, 404)
                return
            try:
                snapshot = request_snapshot(request_id)
            except KeyError:
                snapshot = {
                    "request_id": request_id,
                    "route": "/api/tasks",
                    "current_stage": "task_queue",
                    "current_activity": {
                        "status": str(task["state"]),
                        "summary": "Task saved and queued for execution.",
                    },
                    "progress": 0.02,
                    "live_steps": [],
                }
            # TaskStore is the lifecycle authority.  This endpoint exposes
            # best-effort trace details without publishing a second state.
            snapshot.update({
                "status": str(task["state"]),
                "task_state": str(task["state"]),
                "terminal": bool(task["terminal"]),
                "failure_class": task.get("failure_class"),
                "termination_reason": task.get("termination_reason"),
            })
            self.send_json(snapshot)
            return
        if parsed.path == "/api/config":
            cache_registry = CacheRegistry()
            cache_inventory = cache_registry.inventory()
            cache_telemetry = cache_registry.resolve_telemetry()
            memory_status = CaseMemoryStore(case_memory_path()).status()
            self.send_json({
                "providers": provider_status(),
                "runtime": {
                    **runtime_status(),
                    # Expose the cached and live ceilings used by the single
                    # Agent path. Capabilities perform exact cache lookup
                    # internally when a live request is allowed.
                    "agent_request_budget_seconds": request_budget_seconds("cached"),
                    "live_agent_request_budget_seconds": request_budget_seconds("live"),
                },
                "cache_registry": cache_inventory,
                "cache_resolve_telemetry": cache_telemetry,
                "case_memory": memory_status,
                "mode": "unified",
                "model_routing": {
                    "strategy": "fixed_single_model",
                    "provider": FIXED_PROVIDER_NAME,
                    "model": FIXED_MODEL_NAME,
                    "auto_failover": False,
                },
                "cascast_vil_thresholds": cascast_vil_threshold_payload(),
                "default_cascast_vil_threshold": int(DEFAULT_CASCAST_VIL_THRESHOLD),
            })
            return
        if parsed.path == "/api/sessions":
            sessions = []
            for session_id in available_session_ids():
                state = load_session(session_id)
                turns = state.get("turns", [])
                if not turns:
                    continue
                sessions.append({
                    "id": session_id,
                    "title": str(turns[0].get("question", ""))[:400],
                    "created_at": str(turns[0].get("created_at", "")),
                    "updated_at": str(turns[-1].get("created_at", "")),
                })
            self.send_json({
                "session_ids": [item["id"] for item in sessions],
                "sessions": sessions,
            })
            return
        if parsed.path == "/api/session":
            query = parse_qs(parsed.query)
            requested = str((query.get("session_id") or [""])[0]).strip()
            use_latest = str((query.get("latest") or [""])[0]).lower() in {
                "1", "true", "yes"
            }
            try:
                session_id = normalize_session_id(requested)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
                return
            state = load_session(session_id)
            if use_latest:
                session_id = latest_session_id() or session_id
                state = load_session(session_id)
            self.send_json({
                "session_id": session_id,
                "turn_count_total": len(state.get("turns", [])),
                "study_case": evidence_index(rebuild_study_case(session_id, state.get("turns", []))),
                "turns": [
                    {
                        "turn_id": item.get("turn_id"),
                        "request_id": item.get("request_id", ""),
                        "created_at": item.get("created_at"),
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "selected_tools": item.get("selected_tools", []),
                        "termination_reason": item.get("termination_reason"),
                        "code_execution_count": item.get("code_execution_count", 0),
                        "evidence_execution_count": item.get("evidence_execution_count", 0),
                        "evidence_tool_call_count": item.get("evidence_tool_call_count", 0),
                        "postprocessing_success_count": item.get("postprocessing_success_count", 0),
                        "tool_cost_summary": item.get("tool_cost_summary", {}),
                        "artifacts": session_artifacts(item.get("artifacts", [])),
                    }
                    for item in state.get("turns", [])
                ],
            })
            return
        artifact_match = re.fullmatch(
            r"/api/artifacts/([0-9a-f]{32})\.png", parsed.path
        )
        if artifact_match:
            name = f"{artifact_match.group(1)}.png"
            path = next(
                (root / name for root in artifact_read_roots() if (root / name).is_file()),
                None,
            )
            if path is None:
                self.send_json({"error": "artifact not found"}, 404)
            else:
                self.send_png(path.read_bytes(), immutable=True)
            return
        if parsed.path == "/api/uploads":
            self.send_json(list_uploaded_records())
            return
        if parsed.path == "/api/uploaded-frame":
            try:
                query = parse_qs(parsed.query)
                upload_id = str(query.get("upload_id", [""])[0])
                self.send_png(render_png(
                    uploaded_frame(upload_id),
                    variable_metadata={"missing_value": 255},
                    style="encoded_vil",
                ))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/api/uploaded-sequence-frame":
            try:
                query = parse_qs(parsed.query)
                frame = uploaded_sequence_frame(
                    str(query.get("upload_id", [""])[0]),
                    int(query.get("index", ["0"])[0]),
                )
                self.send_png(render_png(
                    frame,
                    variable_metadata={"missing_value": 255},
                    style="encoded_vil",
                ))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        super().do_GET()

    def do_DELETE(self) -> None:
        if urlparse(self.path).path != "/api/session":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            payload = self._read_payload()
            self.send_json(delete_session_turn(
                normalize_session_id(payload.get("session_id")),
                normalize_turn_id(payload.get("turn_id")),
            ))
        except ValueError as error:
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            try:
                payload = self._read_payload()
            except ValueError as error:
                if path == "/api/tasks" or re.fullmatch(
                    r"/api/tasks/[0-9A-Za-z_-]{8,128}/(cancel|retry)",
                    path,
                ):
                    self._send_task_error(ValueError(str(error)))
                    return
                raise
            if self._handle_task_post(path, payload):
                return
            if path == "/api/upload":
                self.send_json(register_uploaded_vil(
                    filename=str(payload.get("filename", "uploaded_vil.png")),
                    width=int(payload.get("width", 0)),
                    height=int(payload.get("height", 0)),
                    pixels_base64=str(payload.get("pixels_base64", "")),
                    grayscale_source=bool(payload.get("grayscale_source", False)),
                    event_id=str(payload.get("event_id", "")),
                    metadata=payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict) else None,
                ))
                return
            if path == "/api/upload-sequence":
                self.send_json(register_uploaded_vil_sequence(
                    list(payload.get("frames", [])),
                    event_id=str(payload.get("event_id", "")),
                    metadata=payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict) else None,
                ))
                return
            if path == "/api/merge-sessions":
                self.send_json(merge_sessions(
                    payload.get("source_session_ids", []), payload.get("target_session_id"),
                ))
                return
            self.send_json({"error": "not found"}, 404)
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": str(error)}, 500)


class WeatherAgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    # Windows SO_REUSEADDR permits competing listeners on the same endpoint.
    # Own this port exclusively so stale and updated Agents cannot both serve it.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[SimpleHTTPRequestHandler],
        bind_and_activate: bool = True,
        *,
        task_store: TaskStore | None = None,
        task_executor: TaskExecutor | None = None,
        task_worker_count: int = 1,
    ) -> None:
        self.task_store = task_store or TaskStore()
        self.task_execution_available = task_executor is not None
        super().__init__(
            server_address,
            request_handler_class,
            bind_and_activate=bind_and_activate,
        )
        # A runner object is always present so task state/events remain
        # inspectable.  Workers start only with a real injected adapter;
        # otherwise queued work would be accepted and falsely failed by a
        # placeholder implementation.
        self.task_runner = TaskRunner(
            self.task_store,
            task_executor or self._missing_task_executor,
            worker_count=task_worker_count,
            recover_on_start=True,
        )
        if self.task_execution_available:
            self.task_recovery = self.task_runner.start().get("recovery", {})
        else:
            self.task_recovery = self.task_store.recover_startup()

    @staticmethod
    def _missing_task_executor(_context: Any) -> Any:
        raise RuntimeError("asynchronous task executor is not configured")

    def server_close(self) -> None:
        try:
            runner = getattr(self, "task_runner", None)
            if runner is not None:
                runner.stop(wait=True, timeout_seconds=2.0)
            self.task_store.close()
        finally:
            super().server_close()


def main() -> None:
    host = os.getenv("WEATHER_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("WEATHER_AGENT_PORT", "8000"))
    server = WeatherAgentHTTPServer(
        (host, port),
        WeatherAgentRequestHandler,
        task_executor=UnifiedAgentTaskExecutor(ExecutionDependencies(
            providers=PROVIDERS, select_model=fixed_model_selection,
            complete=post_chat_completion,
            summarize=post_context_summary,
        )),
    )
    print(f"Multi-Source Weather Agent: http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
