from __future__ import annotations

"""Background worker threads for the single-host SQLite task queue."""

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from ..persistence.task_store import TaskStore


TASK_QUEUE_API_SCHEMA_VERSION = "task-queue-api-v1"


class TaskExecutionCancelled(RuntimeError):
    """An executor may raise this after observing ``context.should_cancel``."""


@dataclass(frozen=True)
class TaskOutcome:
    """Normalized terminal output returned by an injected executor."""

    state: str = "succeeded"
    result: Mapping[str, Any] = field(default_factory=dict)
    result_ref: str | None = None
    failure_class: str | None = None
    termination_reason: str | None = None
    error: Mapping[str, Any] = field(default_factory=dict)
    turn_commit: Mapping[str, Any] | None = None

    @classmethod
    def from_value(cls, value: TaskOutcome | Mapping[str, Any]) -> TaskOutcome:
        if isinstance(value, cls):
            return value
        record = dict(value)
        explicit_state = str(record.get("task_state") or "").strip().lower()
        status = str(record.get("status") or "").strip().lower()
        if explicit_state:
            state = explicit_state
        elif status in {"cancelled", "canceled"}:
            state = "cancelled"
        elif status == "interrupted":
            state = "interrupted"
        elif status in {
            "error", "failed", "failure", "partial", "unavailable", "blocked",
        }:
            state = "failed"
        else:
            state = "succeeded"
        error = record.get("error")
        if not isinstance(error, Mapping):
            error = {"message": str(error)} if error else {}
        return cls(
            state=state,
            # Preserve the complete bounded executor envelope for a later HTTP
            # result endpoint.  Scientific arrays should live behind refs.
            result=record,
            result_ref=record.get("result_ref"),
            failure_class=record.get("failure_class"),
            termination_reason=record.get("termination_reason"),
            error=dict(error),
            turn_commit=record.get("turn_commit"),
        )


class TaskExecutionContext:
    """Request facts and cooperative cancellation for one queued execution."""

    def __init__(self, store: TaskStore, task: Mapping[str, Any], owner_instance_id: str):
        self.store = store
        self.request_id = str(task["request_id"])
        self.owner_instance_id = owner_instance_id
        self._initial_task = dict(task)

    @property
    def request(self) -> Mapping[str, Any]:
        return self._initial_task["request"]

    def snapshot(self) -> dict[str, Any]:
        return self.store.get_task(self.request_id)

    def should_cancel(self) -> bool:
        task = self.snapshot()
        return task["cancel_requested"] or task["state"] in {"cancelled", "cancelling", "interrupted"}


TaskExecutor = Callable[[TaskExecutionContext], TaskOutcome | Mapping[str, Any]]


class TaskRunner:
    def __init__(self, store: TaskStore, executor: TaskExecutor, *, instance_id=None,
                 worker_count=1, poll_interval_seconds=0.05, recover_on_start=True):
        self.store = store
        self.executor = executor
        self.instance_id = instance_id or f"runner-{uuid.uuid4().hex}"
        self.worker_count = max(1, min(16, int(worker_count)))
        self.poll_interval_seconds = max(0.005, float(poll_interval_seconds))
        self.recover_on_start = recover_on_start
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.RLock()
        self._active_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._active_request_ids = set()
        self._recovery_summary = {}
        self._last_runner_error = None

    def start(self):
        with self._lifecycle_lock:
            if self._threads:
                return self.status()
            self._stop.clear()
            if self.recover_on_start:
                self._recovery_summary = self.store.recover_startup()
            self._threads = [threading.Thread(target=self._worker_loop, daemon=True,
                name=f"{self.instance_id}-worker-{i+1}") for i in range(self.worker_count)]
            for thread in self._threads:
                thread.start()
        return self.status()

    def stop(self, *, wait=True, timeout_seconds=5.0):
        self._stop.set()
        self.wake()
        if wait:
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            for thread in list(self._threads):
                thread.join(timeout=max(0.0, deadline-time.monotonic()))
        with self._lifecycle_lock:
            stopped = not any(t.is_alive() for t in self._threads)
            if stopped:
                self._threads = []
        return stopped

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    @staticmethod
    def _response(task, **extra):
        return {"schema_version": TASK_QUEUE_API_SCHEMA_VERSION, "task": dict(task), **extra}

    def submit_task(self, **arguments):
        task, created = self.store.create_task(**arguments)
        self.wake()
        return self._response(task, created=created)

    def get_task(self, request_id):
        return self._response(self.store.get_task(request_id))

    def get_events(self, request_id, *, after_sequence=0):
        return self._response(self.store.get_task(request_id),
            events=self.store.list_events(request_id, after_sequence=after_sequence))

    def cancel_task(self, request_id, *, requested_by="user"):
        task = self.store.request_cancel(request_id, requested_by=requested_by)
        self.wake()
        return self._response(task)

    def retry_task(self, parent_request_id, **arguments):
        task, created = self.store.create_retry(parent_request_id, **arguments)
        self.wake()
        return self._response(task, created=created)

    def wait_for_terminal(self, request_id, *, timeout_seconds=5.0):
        deadline = time.monotonic() + timeout_seconds
        while True:
            task = self.store.get_task(request_id)
            if task["terminal"]:
                return self._response(task)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"task {request_id} did not finish")
            with self._condition:
                self._condition.wait(min(self.poll_interval_seconds, remaining))

    def status(self):
        with self._active_lock:
            active = sorted(self._active_request_ids)
        with self._lifecycle_lock:
            return {"schema_version": TASK_QUEUE_API_SCHEMA_VERSION, "instance_id": self.instance_id,
                "started": bool(self._threads), "worker_count": self.worker_count,
                "running_worker_threads": sum(t.is_alive() for t in self._threads),
                "active_request_ids": active, "recovery": self._recovery_summary,
                "last_runner_error": self._last_runner_error,
                "hard_model_cancel_supported": False, "atomic_result_commit_supported": True}

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                task = self.store.claim_next(self.instance_id)
                if task is not None:
                    self._execute_claimed(task)
                    continue
            except Exception as error:
                self._last_runner_error = f"{type(error).__name__}: {error}"
            with self._condition:
                self._condition.wait(self.poll_interval_seconds)

    def _execute_claimed(self, task):
        request_id = task["request_id"]
        with self._active_lock:
            self._active_request_ids.add(request_id)
        try:
            try:
                outcome = TaskOutcome.from_value(self.executor(
                    TaskExecutionContext(self.store, task, self.instance_id)))
            except TaskExecutionCancelled:
                outcome = TaskOutcome(state="cancelled", result={"status": "cancelled"})
            except Exception as error:
                outcome = TaskOutcome(state="failed", result={"status": "failed"},
                    failure_class="task_executor_error", termination_reason="task_executor_exception",
                    error={"type": type(error).__name__, "message": str(error)[:2000]})
            try:
                self.store.complete_task(request_id, final_state=outcome.state, result=outcome.result,
                    result_ref=outcome.result_ref, failure_class=outcome.failure_class,
                    termination_reason=outcome.termination_reason, error=outcome.error,
                    turn_commit=outcome.turn_commit, owner_instance_id=self.instance_id)
                self._last_runner_error = None
            except Exception as error:
                self._last_runner_error = f"{type(error).__name__}: {error}"
                self.store.complete_task(request_id, final_state="failed",
                    result={"status": "failed", "answer": "本轮结果保存失败，请重试。"},
                    failure_class="result_commit_failed", termination_reason="result_commit_failed",
                    error={"type": type(error).__name__, "message": str(error)[:2000]},
                    owner_instance_id=self.instance_id)
        finally:
            with self._active_lock:
                self._active_request_ids.discard(request_id)
            self.wake()
