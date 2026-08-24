from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest.mock import patch

from sevir_world.agent.unified import run_unified_agent
from sevir_world.api.server import _watch_client_disconnect
from sevir_world.api.unified_service import (
    bind_request_assets,
    execute_unified_request,
)
from sevir_world.code_execution.client import ExecutionClient


class _FixedExecution:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def execute(self, *_args, **_kwargs) -> dict:
        self.calls += 1
        return dict(self.payload)


class PhaseOneAgentControlTests(unittest.TestCase):
    def test_domain_result_error_counts_as_execution_failure(self) -> None:
        replies = iter([
            "<execute>result = {'status': 'error'}</execute>",
            "<execute>result = {'status': 'error'}</execute>",
        ])
        executor = _FixedExecution({
            "status": "ok",
            "result": {"status": "error", "message": "cache miss"},
            "calls": [],
        })

        result = run_unified_agent(
            {},
            "fake",
            "读取真实数据",
            {},
            lambda *_args: {
                "choices": [{"message": {"content": next(replies)}}]
            },
            data_descs=[],
            selected_tools=["ds"],
            executor=executor,
            max_steps=4,
            max_error_retries=2,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["termination_reason"], "consecutive_execution_errors")
        self.assertEqual(result["code_execution_count"], 0)
        execution_steps = [
            item for item in result["agent_trace"] if item["action"] == "execute"
        ]
        self.assertEqual([item["status"] for item in execution_steps], ["error", "error"])
        self.assertEqual(
            [item["error_type"] for item in execution_steps],
            ["DomainResultError", "DomainResultError"],
        )

    def test_shared_deadline_bounds_the_complete_request(self) -> None:
        observed_timeouts: list[float] = []

        def complete(_provider, _model, _messages, timeout_seconds):
            observed_timeouts.append(timeout_seconds)
            return {"choices": [{"message": {"content": "完成。"}}]}

        with patch.dict(
            "os.environ", {"SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS": "300"}
        ):
            result = execute_unified_request(
                "fake",
                "fake-model",
                "简单问题",
                {
                    "selected_tools": ["ds"],
                    "data_descs": [],
                    "session_id": "phase1-deadline",
                    "_deadline_monotonic": time.monotonic() + 0.5,
                },
                providers={"fake": {}},
                complete=complete,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["request_budget"]["limit_seconds"], 30.0)
        self.assertEqual(len(observed_timeouts), 1)
        self.assertGreater(observed_timeouts[0], 0)
        self.assertLessEqual(observed_timeouts[0], 0.5)

        def late_complete(_provider, _model, _messages, _timeout_seconds):
            time.sleep(0.06)
            return {"choices": [{"message": {"content": "太晚了。"}}]}

        started = time.perf_counter()
        with self.assertRaises(TimeoutError):
            execute_unified_request(
                "fake",
                "fake-model",
                "截止时间测试",
                {
                    "selected_tools": ["ds"],
                    "data_descs": [],
                    "session_id": "phase1-deadline-late",
                    "_deadline_monotonic": time.monotonic() + 0.03,
                },
                providers={"fake": {}},
                complete=late_complete,
            )
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_client_cancellation_stops_agent_and_worker(self) -> None:
        peer, server = socket.socketpair()
        cancelled = threading.Event()
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_client_disconnect,
            args=(server, cancelled, stop),
            daemon=True,
        )
        watcher.start()
        peer.close()
        self.assertTrue(cancelled.wait(1.0))
        stop.set()
        watcher.join(timeout=1.0)
        server.close()

        cancelled.clear()
        timer = threading.Timer(0.2, cancelled.set)
        timer.start()
        started = time.perf_counter()
        try:
            result = run_unified_agent(
                {},
                "fake",
                "执行一个长任务",
                {"_cancelled": cancelled.is_set},
                lambda *_args: {
                    "choices": [{
                        "message": {
                            "content": "<execute>import time\ntime.sleep(5)\nresult = {'status': 'ok'}</execute>"
                        }
                    }]
                },
                data_descs=[],
                selected_tools=["ds"],
                executor=ExecutionClient(timeout_seconds=10, memory_limit_mb=1024),
                max_steps=3,
            )
        finally:
            timer.cancel()

        self.assertLess(time.perf_counter() - started, 2.5)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["termination_reason"], "client_cancelled")
        self.assertEqual(result["code_execution_count"], 0)

    def test_last_valid_observation_survives_final_model_failure(self) -> None:
        model_calls = 0

        def complete(_provider, _model, _messages):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return {
                    "choices": [{
                        "message": {"content": "<execute>result = {'value': 42}</execute>"}
                    }]
                }
            raise TimeoutError("final synthesis timed out")

        result = run_unified_agent(
            {},
            "fake",
            "计算并解释",
            {},
            complete,
            data_descs=[],
            selected_tools=["ds"],
            executor=_FixedExecution({
                "status": "ok",
                "result": {"value": 42},
                "calls": [],
            }),
            max_steps=3,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(
            result["termination_reason"], "model_failure_after_observation"
        )
        self.assertEqual(result["observations"][0]["result"], {"value": 42})
        self.assertEqual(result["code_execution_count"], 1)
        self.assertEqual(result["agent_trace"][-1]["action"], "model")
        self.assertEqual(result["agent_trace"][-1]["status"], "error")

    def test_r190_representative_trace_preserves_real_statuses(self) -> None:
        payload: dict[str, object] = {
            "event_id": "R19062708588472",
            "data_policy": "cached",
            "weather_model": "cascast_live",
        }
        descriptions, _ = bind_request_assets("分析R19062708588472", payload)
        replies = iter([
            """<execute>
case = aurora.case()
result = {
    'status': 'ok',
    'event_id': case.event_id,
    'cache_hit': case.cache.cache_hit,
    'leads': case.evidence.aurora.rollout_leads_hours,
}
</execute>""",
            "<solution>R190缓存证据已读取。</solution>",
        ])

        result = run_unified_agent(
            {},
            "fake",
            "分析R19062708588472",
            dict(payload),
            lambda *_args: {
                "choices": [{"message": {"content": next(replies)}}]
            },
            data_descs=descriptions,
            selected_tools=["ds", "aurora"],
            executor=ExecutionClient(timeout_seconds=20, memory_limit_mb=2048),
            max_steps=3,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["termination_reason"], "agent_completed")
        self.assertEqual(result["observations"][0]["status"], "ok")
        self.assertTrue(result["observations"][0]["result"]["cache_hit"])
        self.assertEqual(result["observations"][0]["result"]["leads"], [6, 12, 18, 24])
        self.assertEqual(
            [item["action"] for item in result["agent_trace"]],
            ["execute", "finish"],
        )


if __name__ == "__main__":
    unittest.main()
