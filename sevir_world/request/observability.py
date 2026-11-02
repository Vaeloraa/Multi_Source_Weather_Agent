from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sevir_world.errors import classify_failure, failure_guidance
from .response_pipeline import _public_step


_ROOT = Path(
    os.getenv(
        "SEVIR_OBSERVABILITY_DIR",
        Path(__file__).resolve().parents[2] / "data" / "runtime" / "observability",
    )
)
_LOCK = threading.RLock()
_REQUESTS: dict[str, dict[str, Any]] = {}
_RECENT: deque[dict[str, Any]] = deque(maxlen=1_000)


def _publish_snapshot(snapshot: dict[str, Any], *, recent: bool = False) -> None:
    with _LOCK:
        _REQUESTS[snapshot["request_id"]] = snapshot
        if len(_REQUESTS) > 1_000:
            _REQUESTS.pop(next(iter(_REQUESTS)))
        if recent:
            _RECENT.append(snapshot)



class RequestTrace:
    def __init__(self, request_id: str, route: str = "") -> None:
        self.request_id = str(request_id)
        self._started = time.perf_counter()
        self._record: dict[str, Any] = {
            "request_id": self.request_id,
            "route": route,
            "status": "running",
            "terminal": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": [],
            "failure_class": None,
            "cache_hits": 0,
            "tool_calls": 0,
            "model_calls": 0,
            "current_stage": "request_received",
            "current_activity": {
                "status": "running",
                "summary": "Request received. Preparing context.",
            },
            "progress": 0.01,
            "live_steps": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _publish_snapshot(self.snapshot())

    def add(self, **fields: Any) -> None:
        with _LOCK:
            self._record.update(fields)
            self._record["updated_at"] = datetime.now(timezone.utc).isoformat()
            snapshot = self.snapshot()
        _publish_snapshot(snapshot)

    def update_activity(
        self,
        *,
        stage: str,
        summary: str,
        progress: float | None = None,
        attempt: int | None = None,
        action: str | None = None,
        status: str = "running",
        **metadata: Any,
    ) -> None:
        """Publish the currently running action without waiting for it to finish."""

        activity = {
            "stage": str(stage),
            "summary": str(summary),
            "status": str(status),
            **metadata,
        }
        if attempt is not None:
            activity["attempt"] = int(attempt)
        if action:
            activity["action"] = str(action)
        fields: dict[str, Any] = {
            "current_stage": str(stage),
            "current_activity": activity,
        }
        if progress is not None:
            fields["progress"] = min(0.99, max(0.0, float(progress)))
        self.add(**fields)

    def append_live_step(self, step: dict[str, Any], *, progress: float | None = None) -> None:
        """Append one bounded, browser-safe completed Agent step."""

        public = _public_step(step)
        with _LOCK:
            steps = list(self._record.get("live_steps") or [])
            steps.append(public)
            self._record["live_steps"] = steps[-32:]
            self._record["current_activity"] = {
                **public,
                "status": public.get("status", "ok"),
            }
            if progress is not None:
                self._record["progress"] = min(0.99, max(0.0, float(progress)))
            self._record["updated_at"] = datetime.now(timezone.utc).isoformat()
            snapshot = self.snapshot()
        _publish_snapshot(snapshot)

    def record_stage(self, name: str, elapsed_ms: float, status: str = "ok", **metadata: Any) -> None:
        stage = {
            "stage_id": f"{self.request_id}:{len(self._record['stages']) + 1}",
            "name": name,
            "status": status,
            "elapsed_ms": round(float(elapsed_ms), 2),
            **metadata,
        }
        self._record["stages"].append(stage)
        _publish_snapshot(self.snapshot())

    def finish(self, status: str, error: BaseException | None = None, **fields: Any) -> dict[str, Any]:
        self._record.update(fields)
        self._record["status"] = status
        self._record["terminal"] = True
        self._record["elapsed_ms"] = round((time.perf_counter() - self._started) * 1_000, 2)
        self._record["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            self._record["failure_class"] = self._record.get("failure_class") or classify_failure(error)
            self._record["error"] = {"type": type(error).__name__, "message": str(error)[:1_000]}
        failure_class = self._record.get("failure_class")
        if failure_class:
            cause, next_action = failure_guidance(str(failure_class))
            self._record.setdefault("failure_message", cause)
            self._record.setdefault("next_action", next_action)
        final_ok = str(status).lower() in {"ok", "completed", "success"}
        self._record["current_stage"] = "completed" if final_ok else "failed"
        self._record["progress"] = 1.0
        self._record["current_activity"] = {
            "status": "ok" if final_ok else str(status),
            "summary": "Analysis complete." if final_ok else (
                self._record.get("failure_message") or "Analysis incomplete. See the failed step for details."
            ),
        }
        self._record["updated_at"] = datetime.now(timezone.utc).isoformat()
        snapshot = self.snapshot()
        _publish_snapshot(snapshot, recent=True)
        try:
            _ROOT.mkdir(parents=True, exist_ok=True)
            with (_ROOT / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # Observability must never take down a successful analysis.
            pass
        return deepcopy(snapshot)

    def snapshot(self) -> dict[str, Any]:
        record = deepcopy(self._record)
        if record.get("status") == "running":
            record["elapsed_ms"] = round((time.perf_counter() - self._started) * 1_000, 2)
        return record


def begin_request(request_id: str, route: str = "") -> RequestTrace:
    return RequestTrace(request_id, route)


def request_snapshot(request_id: str) -> dict[str, Any]:
    with _LOCK:
        item = _REQUESTS.get(str(request_id))
    if item is None:
        raise KeyError(request_id)
    snapshot = deepcopy(item)
    if not snapshot["terminal"]:
        started = datetime.fromisoformat(snapshot["started_at"])
        snapshot["elapsed_ms"] = round(
            (datetime.now(timezone.utc) - started).total_seconds() * 1_000, 2,
        )
    return snapshot


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
    return {
        "status": "ok",
        "scope": "process",
        "window_requests": len(items),
        "latency_ms": {"p50": _percentile(elapsed, 0.50), "p95": _percentile(elapsed, 0.95)},
        "failure_classes": dict(failures),
        "model_calls": sum(int(item.get("model_calls", 0)) for item in items),
        "tool_calls": sum(int(item.get("tool_calls", 0)) for item in items),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in items),
    }
