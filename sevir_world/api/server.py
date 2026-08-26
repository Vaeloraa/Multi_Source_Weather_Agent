from __future__ import annotations

"""Thin HTTP boundary for the single unified Agent."""

import hashlib
import io
import json
import os
import re
import select
import socket
import tempfile
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from sevir_world.agent.session import (
    append_turn,
    available_session_ids,
    delete_turn,
    latest_session_id,
    load_session,
    normalize_session_id,
    normalize_turn_id,
)
from sevir_world.api.model_client import (
    FIXED_MODEL_NAME,
    FIXED_PROVIDER_NAME,
    PROVIDERS,
    fixed_model_selection,
    post_chat_completion,
    provider_status,
)
from sevir_world.api.observability import (
    begin_request,
    metrics_snapshot,
    request_snapshot,
    set_current_trace,
)
from sevir_world.api.response_pipeline import finalize_agent_response
from sevir_world.api.status import runtime_status
from sevir_world.api.unified_service import (
    bind_request_assets,
    execute_unified_request,
    request_budget_seconds,
)
from sevir_world.agent.unified import AgentCancelled
from sevir_world.agent.study import (
    apply_study_case_context,
    evidence_index,
    should_inherit_study_case_context,
)
from sevir_world.tools.geolocator.Geospatial import resolve_place
from sevir_world.data_access.ensemble import get_cascast_display_frame
from sevir_world.data_access.request_context import enrich_request_context
from sevir_world.data_access.uploads import (
    get_uploaded_display_frame,
    get_uploaded_sequence_display_frame,
    list_uploaded_records,
    register_uploaded_vil,
    register_uploaded_vil_sequence,
)
from sevir_world.domain.thresholds import (
    DEFAULT_CASCAST_VIL_THRESHOLD,
    cascast_vil_threshold_payload,
    normalize_cascast_vil_threshold,
)
from sevir_world.domain.vil_palette import colorize_vil


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
_DEFAULT_ARTIFACT_ROOT = ROOT / "data" / "artifacts"
_FALLBACK_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "sevir_world_artifacts"


def _artifact_root() -> Path:
    # The isolated Windows worker can receive mojibake for a non-ASCII repo
    # path even when the HTTP process itself can write there. Use the
    # ASCII-safe runtime directory by default; an explicit operator setting
    # still wins. Legacy workspace artifacts remain readable below.
    candidate = Path(os.getenv("SEVIR_ARTIFACT_DIR", str(_FALLBACK_ARTIFACT_ROOT)))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        _FALLBACK_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        return _FALLBACK_ARTIFACT_ROOT


ARTIFACT_ROOT = _artifact_root()
os.environ["SEVIR_ARTIFACT_DIR"] = str(ARTIFACT_ROOT)


def artifact_read_roots() -> tuple[Path, ...]:
    roots = [ARTIFACT_ROOT, _DEFAULT_ARTIFACT_ROOT, _FALLBACK_ARTIFACT_ROOT]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def render_vil_png(frame: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(colorize_vil(frame), mode="RGB").save(
        output, format="PNG", optimize=True
    )
    return output.getvalue()


def render_scalar_png(frame: np.ndarray) -> bytes:
    values = np.asarray(frame, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size:
        lower, upper = np.percentile(finite, [2, 98])
        scale = max(float(upper - lower), 1e-6)
        normalized = np.clip((values - lower) / scale, 0.0, 1.0)
    else:
        normalized = np.zeros_like(values)
    rgb = np.stack(
        [
            (normalized * 255),
            ((1.0 - np.abs(normalized - 0.5) * 2.0) * 210),
            ((1.0 - normalized) * 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def _safe_failure(error: BaseException, request_id: str = "") -> dict[str, Any]:
    return {
        "status": "error",
        "answer": "Agent本轮未完成，请查看步骤日志中的失败位置。",
        "request_id": request_id,
        "route_mode": "unified_agent_failed",
        "termination_reason": "agent_execution_failed",
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:1200],
        },
    }


def _watch_client_disconnect(
    connection: socket.socket,
    cancelled: threading.Event,
    stop: threading.Event,
) -> None:
    """Mark a request cancelled when its HTTP peer closes the socket."""

    while not stop.wait(0.05):
        try:
            readable, _, _ = select.select([connection], [], [], 0)
            if not readable:
                continue
            value = connection.recv(1, socket.MSG_PEEK)
            if not value:
                cancelled.set()
                return
        except (OSError, ValueError):
            # A closed/invalid request socket is equivalent to a disconnected
            # client.  The stop event distinguishes normal handler cleanup.
            if not stop.is_set():
                cancelled.set()
            return


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {format % args}")

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
            raise ValueError("请求过大；处理后的上传数据不得超过5MB。")
        value = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if not isinstance(value, dict):
            raise ValueError("请求体必须是JSON对象。")
        return value

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            self.send_json(metrics_snapshot())
            return
        request_match = re.fullmatch(
            r"/api/requests/([0-9A-Za-z_-]{8,128})", parsed.path
        )
        if request_match:
            try:
                self.send_json(request_snapshot(request_match.group(1)))
            except KeyError:
                self.send_json({"error": "request not found"}, 404)
            return
        if parsed.path == "/api/config":
            self.send_json({
                "providers": provider_status(),
                "runtime": {
                    **runtime_status(),
                    "agent_request_budget_seconds": request_budget_seconds(),
                },
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
            if use_latest or (not state.get("turns") and latest_session_id()):
                session_id = latest_session_id() or session_id
                state = load_session(session_id)
            self.send_json({
                "session_id": session_id,
                "study_case": evidence_index(state.get("study_case")),
                "turns": [
                    {
                        "turn_id": item.get("turn_id"),
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
                        "artifacts": item.get("artifacts", []),
                    }
                    for item in state.get("turns", [])[-30:]
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
                self.send_png(render_vil_png(get_uploaded_display_frame(upload_id)))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/api/uploaded-sequence-frame":
            try:
                query = parse_qs(parsed.query)
                frame = get_uploaded_sequence_display_frame(
                    str(query.get("upload_id", [""])[0]),
                    str(query.get("series", ["history"])[0]),
                    int(query.get("index", ["0"])[0]),
                    str(query.get("model_name", ["persistence_baseline"])[0]),
                    int(query["lead"][0]) if query.get("lead", [""])[0] else None,
                )
                series = str(query.get("series", ["history"])[0])
                self.send_png(
                    render_scalar_png(frame)
                    if series.startswith("uncertainty_")
                    else render_vil_png(frame)
                )
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/api/cascast-frame":
            try:
                query = parse_qs(parsed.query)
                series = str(query.get("series", ["forecast"])[0])
                frame, _ = get_cascast_display_frame(
                    event_id=str(query.get("event_id", [""])[0]),
                    series=series,
                    frame_index=int(query.get("index", ["0"])[0]),
                    model_name=str(query.get("model_name", ["cascast"])[0]),
                    threshold=normalize_cascast_vil_threshold(
                        query.get(
                            "threshold",
                            [str(int(DEFAULT_CASCAST_VIL_THRESHOLD))],
                        )[0]
                    ),
                    lead_index=(
                        int(query["lead"][0])
                        if query.get("lead", [""])[0] else None
                    ),
                )
                self.send_png(
                    render_scalar_png(frame)
                    if series.startswith("uncertainty_") or "probability" in series
                    else render_vil_png(frame)
                )
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
            self.send_json(delete_turn(
                normalize_session_id(payload.get("session_id")),
                normalize_turn_id(payload.get("turn_id")),
                str(payload.get("question", "")),
                str(payload.get("answer", "")),
            ))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        trace = None
        request_id = ""
        cancelled = threading.Event()
        cancel_watch_stop = threading.Event()
        cancel_watch: threading.Thread | None = None
        try:
            payload = self._read_payload()
            path = urlparse(self.path).path
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
                source_ids = [
                    normalize_session_id(item)
                    for item in payload.get("source_session_ids", [])
                ]
                target = normalize_session_id(payload.get("target_session_id"))
                if len(source_ids) < 2 or target in source_ids:
                    raise ValueError("合并至少需要两个不同的源对话。")
                turns = [
                    turn
                    for source in source_ids
                    for turn in load_session(source).get("turns", [])
                ]
                turns.sort(key=lambda item: str(item.get("created_at", "")))
                state = {"turns": []}
                for turn in turns:
                    state = append_turn(target, turn)
                self.send_json({
                    "status": "ok",
                    "session_id": target,
                    "turn_count": len(state.get("turns", [])),
                })
                return
            if path != "/api/run":
                self.send_json({"error": "not found"}, 404)
                return

            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("问题不能为空。")
            request_id = str(payload.get("request_id") or uuid.uuid4().hex)
            request_started = time.perf_counter()
            payload["_request_received_monotonic"] = time.monotonic()
            preflight_deadline = time.monotonic() + request_budget_seconds()
            # The same admission deadline must flow through the Agent and
            # response pipeline; otherwise downstream stages silently lose
            # the request budget and can run without a real deadline.
            payload["_deadline_monotonic"] = preflight_deadline
            payload["_cancelled"] = cancelled.is_set
            cancel_watch = threading.Thread(
                target=_watch_client_disconnect,
                args=(self.connection, cancelled, cancel_watch_stop),
                daemon=True,
            )
            cancel_watch.start()
            trace = begin_request(request_id, path)
            set_current_trace(trace)
            trace.add(current_stage="request_received", progress=0.03)

            session_id = normalize_session_id(payload.get("session_id"))
            payload["session_id"] = session_id
            session_state = load_session(session_id)
            active_upload = session_state.get("active_upload", {})
            has_upload = bool(
                str(payload.get("upload_id", ""))
                or str(payload.get("sequence_upload_id", ""))
            )
            if not has_upload and isinstance(active_upload, dict):
                payload.update({
                    key: active_upload.get(key, payload.get(key, ""))
                    for key in ("upload_id", "sequence_upload_id", "data_source")
                })

            trace.add(current_stage="asset_binding", progress=0.08)
            if cancelled.is_set():
                raise AgentCancelled("client disconnected during asset binding")
            payload["context_resolution"] = enrich_request_context(question, payload)
            if payload["context_resolution"].get("status") == "conflict":
                raise ValueError(
                    "请求时间与问题中的时间不一致；请只提供一个明确的UTC时间。"
                )
            if payload["context_resolution"].get("status") == "invalid":
                field = str(payload["context_resolution"].get("field") or "request")
                raise ValueError(f"请求字段 {field} 无效。")
            if (
                payload["context_resolution"].get("status") == "unresolved"
                and str(payload.get("place_name", "")).strip()
            ):
                remaining = preflight_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("地点解析前已达到请求时间上限")
                place = resolve_place(
                    str(payload["place_name"]),
                    timeout_seconds=min(8.0, max(0.1, remaining)),
                )
                payload["place_resolution"] = place
                if place.get("status") == "ok":
                    payload["latitude"] = float(place["latitude"])
                    payload["longitude"] = float(place["longitude"])
            inherit_study_context = should_inherit_study_case_context(
                question, payload, session_state
            )
            payload["_study_context_active"] = inherit_study_context
            if inherit_study_context:
                study_case = session_state.get("study_case", {})
                if isinstance(study_case, dict) and study_case.get("case_id"):
                    payload.setdefault("study_case_id", str(study_case["case_id"]))
                apply_study_case_context(payload, session_state)
            bind_request_assets(question, payload)
            if not inherit_study_context and not str(payload.get("study_case_id") or "").strip():
                # A self-contained question starts an isolated evidence scope,
                # while remaining visible in the same conversation history.
                scope_token = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{session_id}:{request_id}"
                ).hex[:16]
                payload["study_case_id"] = f"{session_id}-{scope_token}"
            # Client-supplied choices are intentionally ignored. The current
            # demo uses one fixed model and returns its first error without
            # probing fallback candidates.
            provider_name, model = fixed_model_selection(PROVIDERS)
            payload["provider"] = provider_name
            payload["model"] = model

            trace.add(current_stage="unified_agent", progress=0.15)
            analysis_started = time.perf_counter()
            result = execute_unified_request(
                provider_name,
                model,
                question,
                payload,
                providers=PROVIDERS,
                complete=lambda provider, selected_model, messages, timeout_seconds: post_chat_completion(
                    provider,
                    selected_model,
                    messages,
                    timeout_seconds=min(
                        timeout_seconds,
                        float(os.getenv("MODEL_API_TIMEOUT_SECONDS", "30")),
                    ),
                ),
            )
            result = finalize_agent_response(
                result,
                payload=payload,
                question=question,
                request_id=request_id,
                mode="unified",
                provider_name=provider_name,
                model_name=model,
                route_decision={
                    "requested_mode": "unified",
                    "selected_mode": "unified",
                    "reason": "single_agent_path",
                },
                started=request_started,
                analysis_started=analysis_started,
                trace=trace,
                presentation={
                    "event_id": str(payload.get("event_id", "")),
                    "requested_visualization": bool(
                        payload.get("visual_artifact_required")
                        or re.search(r"图|chart|heatmap|bar", question, re.IGNORECASE)
                    ),
                },
                artifact_root=ARTIFACT_ROOT,
                deadline_monotonic=float(payload.get("_deadline_monotonic", time.monotonic())),
                cancelled=cancelled.is_set,
            )
            if not cancelled.is_set():
                self.send_json(result)
        except AgentCancelled as error:
            if trace is not None:
                trace.finish("cancelled", error)
            if not cancelled.is_set():
                self.send_json({
                    "status": "cancelled",
                    "answer": "请求已取消。",
                    "request_id": request_id,
                    "termination_reason": "client_cancelled",
                }, 408)
        except (ValueError, RuntimeError, TimeoutError, KeyError, IndexError,
                json.JSONDecodeError) as error:
            failure = _safe_failure(error, request_id)
            if trace is not None:
                failure["request_trace"] = trace.finish("error", error)
            if not cancelled.is_set():
                self.send_json(failure, 400 if isinstance(error, ValueError) else 503)
        except Exception as error:
            failure = _safe_failure(error, request_id)
            if trace is not None:
                failure["request_trace"] = trace.finish("error", error)
            if not cancelled.is_set():
                self.send_json(failure, 500)
        finally:
            cancel_watch_stop.set()
            if cancel_watch is not None:
                cancel_watch.join(timeout=0.2)


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


def main() -> None:
    host = os.getenv("SEVIR_DEMO_HOST", "127.0.0.1")
    port = int(os.getenv("SEVIR_DEMO_PORT", "8000"))
    server = DemoHTTPServer((host, port), DemoHandler)
    print(f"SEVIR Agent: http://{host}:{port}")
    print("按Ctrl+C停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    main()
