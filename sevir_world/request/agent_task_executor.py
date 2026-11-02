from __future__ import annotations

"""Adapter from durable TaskRunner attempts to the one real Agent boundary."""

import time

from sevir_world.agent.contracts import ExecutionRuntime
from sevir_world.agent.unified import AgentCancelled
from sevir_world.request.observability import begin_request
from sevir_world.request.service import (
    ExecutionDependencies,
    execute_request,
)
from sevir_world.request.task_runner import (
    TaskExecutionCancelled,
    TaskExecutionContext,
    TaskOutcome,
)
from .task_requests import TASK_REQUEST_ENVELOPE_SCHEMA_VERSION


class UnifiedAgentTaskExecutor:
    """Replay one durable raw request through the transport-independent boundary."""

    def __init__(self, dependencies: ExecutionDependencies) -> None:
        self.dependencies = dependencies

    def __call__(self, context: TaskExecutionContext) -> TaskOutcome:
        task = context.snapshot()
        envelope = context.request
        schema = envelope["schema_version"]
        if schema != TASK_REQUEST_ENVELOPE_SCHEMA_VERSION:
            return TaskOutcome(
                state="failed",
                result={
                    "status": "failed",
                    "failure_class": "task_envelope_schema_mismatch",
                    "termination_reason": "unsupported_task_request_envelope",
                    "expected_schema_version": TASK_REQUEST_ENVELOPE_SCHEMA_VERSION,
                    "actual_schema_version": schema,
                },
                failure_class="task_envelope_schema_mismatch",
                termination_reason="unsupported_task_request_envelope",
            )
        if context.should_cancel():
            raise TaskExecutionCancelled("task cancelled before request replay")
        # The durable envelope stores caller input, not a transport-prepared
        # business snapshot.  ``execute_request`` is the sole owner of request
        # context enrichment, study inheritance and asset binding.
        payload = dict(envelope["payload"])
        question = payload["question"]
        # The task row—not a copied parent payload—is authoritative for every
        # attempt identity and for the confirmation committed by TaskStore.
        payload.update({
            "request_id": context.request_id,
            "session_id": str(task["session_id"]),
            "turn_id": str(task["turn_id"]),
        })
        deadline_epoch = (
            float(envelope["deadline_at_epoch"]) if int(task["attempt_number"]) == 1
            else float(task["created_at"]) + float(envelope["budget_seconds"])
        )
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            return TaskOutcome(
                state="failed",
                result={
                    "status": "failed",
                    "request_id": context.request_id,
                    "failure_class": "queue_deadline_expired",
                    "termination_reason": "task_deadline_expired_before_execution",
                },
                failure_class="queue_deadline_expired",
                termination_reason="task_deadline_expired_before_execution",
            )
        started = time.perf_counter()
        deadline_monotonic = time.monotonic() + remaining
        trace = begin_request(context.request_id, "/api/tasks")
        runtime = ExecutionRuntime(
            request_id=context.request_id,
            session_id=str(task["session_id"]),
            turn_id=str(task["turn_id"]),
            deadline_monotonic=deadline_monotonic,
            budget_seconds=remaining,
            cancelled=context.should_cancel,
            trace=trace,
            started_perf_counter=started,
        )
        try:
            return execute_request(
                question=question,
                payload=payload,
                runtime=runtime,
                dependencies=self.dependencies,
            )
        except AgentCancelled as error:
            trace.finish("cancelled", error)
            raise TaskExecutionCancelled(str(error)) from error
        except Exception as error:
            trace.finish("failed", error)
            raise
