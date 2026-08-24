from __future__ import annotations

import json
import math
import os
import threading
import time
import contextvars
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_ROOT = Path(
    os.getenv(
        "SEVIR_OBSERVABILITY_DIR",
        Path(__file__).resolve().parents[2] / "data" / "runtime" / "observability",
    )
)
_LOCK = threading.RLock()
_REQUESTS: dict[str, dict[str, Any]] = {}
_RECENT: deque[dict[str, Any]] = deque(maxlen=1_000)
_CURRENT_TRACE: contextvars.ContextVar["RequestTrace | None"] = contextvars.ContextVar("sevir_current_trace", default=None)
_TRACE_LOCAL = threading.local()


def classify_failure(error: BaseException | str, stage: str = "") -> str:
    text = str(error).lower()
    stage_text = str(stage).lower()
    if "circuit" in text or "熔断" in text:
        return "transport_circuit_open"
    if "429" in text or "rate limit" in text or "限流" in text:
        return "provider_rate_limit"
    if any(code in text for code in ("http 500", "http 502", "http 503", "http 504", "http 5")):
        return "provider_5xx"
    if "connect" in text and ("timeout" in text or "timed out" in text or "超时" in text):
        return "transport_connect_timeout"
    if "timeout" in text or "timed out" in text or "未返回" in text:
        return "execution_timeout" if "execution" in stage_text or "worker" in stage_text else "transport_read_timeout"
    if any(marker in text for marker in ("reset", "remote disconnected", "远端中断", "broken pipe")):
        return "transport_reset"
    if "budget" in text or "预算" in text:
        return "analysis_budget_exhausted"
    if "render" in stage_text or "png" in text or "visual" in text:
        return "render_error"
    if "argument" in text or "参数" in text:
        return "tool_argument_error"
    if "tool" in stage_text:
        return "tool_runtime_error"
    if "disconnect" in text or "client" in stage_text:
        return "client_disconnected"
    if "json" in text or "protocol" in text or "协议" in text:
        return "model_protocol_error"
    return "unknown_error"


class RequestTrace:
    def __init__(self, request_id: str, route: str = "") -> None:
        self.request_id = str(request_id)
        self._started = time.perf_counter()
        self._record: dict[str, Any] = {
            "request_id": self.request_id,
            "route": route,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": [],
            "failure_class": None,
            "fallback": False,
            "cache_hits": 0,
            "tool_calls": 0,
            "model_calls": 0,
        }
        with _LOCK:
            _REQUESTS[self.request_id] = self.snapshot()

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        stage = {"stage_id": f"{self.request_id}:{len(self._record['stages']) + 1}", "name": name, **metadata}
        try:
            yield stage
            stage["status"] = "ok"
        except BaseException as error:
            stage["status"] = "error"
            stage["failure_class"] = classify_failure(error, name)
            stage["error"] = {"type": type(error).__name__, "message": str(error)[:1_000]}
            self._record["failure_class"] = stage["failure_class"]
            raise
        finally:
            stage["elapsed_ms"] = round((time.perf_counter() - started) * 1_000, 2)
            self._record["stages"].append(stage)
            with _LOCK:
                _REQUESTS[self.request_id] = self.snapshot()

    def add(self, **fields: Any) -> None:
        self._record.update(fields)
        with _LOCK:
            _REQUESTS[self.request_id] = self.snapshot()

    def record_stage(self, name: str, elapsed_ms: float, status: str = "ok", **metadata: Any) -> None:
        stage = {
            "stage_id": f"{self.request_id}:{len(self._record['stages']) + 1}",
            "name": name,
            "status": status,
            "elapsed_ms": round(float(elapsed_ms), 2),
            **metadata,
        }
        self._record["stages"].append(stage)
        with _LOCK:
            _REQUESTS[self.request_id] = self.snapshot()

    def increment(self, field: str, amount: int = 1) -> None:
        self._record[field] = int(self._record.get(field, 0)) + int(amount)

    def finish(self, status: str, error: BaseException | None = None, **fields: Any) -> dict[str, Any]:
        self._record.update(fields)
        self._record["status"] = status
        self._record["elapsed_ms"] = round((time.perf_counter() - self._started) * 1_000, 2)
        self._record["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            self._record["failure_class"] = self._record.get("failure_class") or classify_failure(error)
            self._record["error"] = {"type": type(error).__name__, "message": str(error)[:1_000]}
        snapshot = self.snapshot()
        with _LOCK:
            _REQUESTS[self.request_id] = snapshot
            _RECENT.append(snapshot)
            try:
                _ROOT.mkdir(parents=True, exist_ok=True)
                with (_ROOT / "requests.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
            except OSError:
                # Observability must never take down a successful analysis.
                pass
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._record, ensure_ascii=False, default=str))


def begin_request(request_id: str, route: str = "") -> RequestTrace:
    return RequestTrace(request_id, route)


def set_current_trace(trace: RequestTrace | None) -> contextvars.Token:
    _TRACE_LOCAL.trace = trace
    return _CURRENT_TRACE.set(trace)


def reset_current_trace(token: contextvars.Token) -> None:
    _CURRENT_TRACE.reset(token)


def current_trace() -> RequestTrace | None:
    return getattr(_TRACE_LOCAL, "trace", None) or _CURRENT_TRACE.get()


def request_snapshot(request_id: str) -> dict[str, Any]:
    with _LOCK:
        item = _REQUESTS.get(str(request_id))
        if item is None:
            raise KeyError(request_id)
        return json.loads(json.dumps(item, ensure_ascii=False, default=str))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 2)


def metrics_snapshot() -> dict[str, Any]:
    with _LOCK:
        items = list(_RECENT)
    elapsed = [float(item.get("elapsed_ms", 0)) for item in items]
    failures = Counter(str(item.get("failure_class")) for item in items if item.get("failure_class"))
    fallbacks = sum(1 for item in items if item.get("fallback"))
    return {
        "status": "ok",
        "window_requests": len(items),
        "latency_ms": {"p50": _percentile(elapsed, 0.50), "p95": _percentile(elapsed, 0.95)},
        "fallback_rate": round(fallbacks / len(items), 4) if items else 0.0,
        "failure_classes": dict(failures),
        "model_calls": sum(int(item.get("model_calls", 0)) for item in items),
        "tool_calls": sum(int(item.get("tool_calls", 0)) for item in items),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in items),
    }
