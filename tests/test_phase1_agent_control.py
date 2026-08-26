from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest.mock import patch

from sevir_world.agent.unified import (
    _action_message,
    _observation_for_model,
    _parse_action,
    _solution_artifact_error,
    SYSTEM_PROMPT,
    run_unified_agent,
)
from sevir_world.agent.contracts import ExecutionRequest
from sevir_world.api.server import _watch_client_disconnect
from sevir_world.api.unified_service import (
    bind_request_assets,
    execute_unified_request,
    request_budget_seconds,
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
    def test_forecast_prompt_never_allows_observation_fallback(self) -> None:
        self.assertIn("only frames returned by ``cascast.forecast``", SYSTEM_PROMPT)
        self.assertIn("never a fallback forecast", SYSTEM_PROMPT)
        self.assertIn("cache_only_miss", SYSTEM_PROMPT)

    def test_reflective_action_preserves_the_models_analysis_preamble(self) -> None:
        message = _action_message(
            "先按预测前资料筛选样本。\n<execute>result = {'n': 5}",
            "execute",
            "result = {'n': 5}",
        )
        self.assertIn("先按预测前资料筛选样本", message)
        self.assertTrue(message.endswith("</execute>"))

    def test_nested_observation_values_and_artifact_links_remain_grounded(self) -> None:
        observation = _observation_for_model({
            "status": "ok",
            "result": {
                "comparison": {
                    "history": {
                        "metrics": {
                            "movement": {"mean": 2.5, "sample_count": 5}
                        }
                    }
                }
            },
        })
        self.assertEqual(
            observation["result"]["comparison"]["history"]["metrics"]["movement"]["mean"],
            2.5,
        )
        artifact = {"url": "/api/artifacts/" + "a" * 32 + ".png"}
        self.assertIsNone(_solution_artifact_error(
            "![图](/api/artifacts/" + "a" * 32 + ".png)", [artifact]
        ))
        self.assertIn("unregistered", _solution_artifact_error(
            "![图](/api/artifacts/" + "b" * 32 + ".png)", [artifact]
        ) or "")
        self.assertIn("malformed", _solution_artifact_error(
            "![图](/api/artifacts/chart.png)", [artifact]
        ) or "")
        self.assertIsNone(_solution_artifact_error(
            "候选地点已经列出。", [], required=False
        ))
        self.assertIn("explicitly requested", _solution_artifact_error(
            "这是文字结论。", [], required=True
        ) or "")

    def test_agent_prompt_exposes_question_level_visual_contract(self) -> None:
        prompts: list[list[dict[str, str]]] = []

        def complete(_provider, _model, messages):
            prompts.append([dict(item) for item in messages])
            return {"choices": [{"message": {"content": "<solution>请确认具体城市。</solution>"}}]}

        result = run_unified_agent(
            {}, "fake", "请列出Springfield候选并要求确认。", {}, complete,
            data_descs=[], selected_tools=["ds"], executor=_FixedExecution({}),
            max_steps=1,
        )
        self.assertEqual(result["status"], "ok")
        first_prompt = "\n".join(item["content"] for item in prompts[0])
        self.assertIn("Visual artifact contract:\nrequired=false", first_prompt)
        self.assertIn("A figure is optional", first_prompt)

    def test_required_visual_can_repair_an_early_text_only_solution(self) -> None:
        artifact_url = "/api/artifacts/" + "c" * 32 + ".png"
        prompts: list[list[dict[str, str]]] = []
        replies = iter([
            "<solution>先给文字结论。</solution>",
            "<execute>result = {'figure': 'saved'}</execute>",
            f"<solution>结论。![图]({artifact_url})</solution>",
        ])

        def complete(_provider, _model, messages):
            prompts.append([dict(item) for item in messages])
            return {"choices": [{"message": {"content": next(replies)}}]}

        result = run_unified_agent(
            {}, "fake", "请生成一张变化曲线图。", {}, complete,
            data_descs=[], selected_tools=["ds"],
            executor=_FixedExecution({
                "status": "ok",
                "result": {"figure": "saved"},
                "calls": [],
                "generated_artifacts": [{"url": artifact_url}],
            }),
            max_steps=3,
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("Return one <execute> program now", prompts[1][-1]["content"])
        self.assertEqual(result["artifacts"][0]["url"], artifact_url)

    def test_request_fact_can_require_visual_without_visual_words(self) -> None:
        artifact_url = "/api/artifacts/" + "d" * 32 + ".png"
        prompts: list[list[dict[str, str]]] = []
        replies = iter([
            "<execute>result = {'figure': 'saved'}</execute>",
            f"<solution>区域判断。![证据图]({artifact_url})</solution>",
        ])

        def complete(_provider, _model, messages):
            prompts.append([dict(item) for item in messages])
            return {"choices": [{"message": {"content": next(replies)}}]}

        result = run_unified_agent(
            {}, "fake", "请用Aurora做区域判断。",
            {}, complete,
            execution_request=ExecutionRequest.from_payload(
                [], ["ds"], visual_artifact_required=True
            ),
            executor=_FixedExecution({
                "status": "ok",
                "result": {"figure": "saved"},
                "calls": [],
                "generated_artifacts": [{"url": artifact_url}],
            }),
            max_steps=2,
        )

        first_prompt = "\n".join(item["content"] for item in prompts[0])
        self.assertIn("Visual artifact contract:\nrequired=true", first_prompt)
        self.assertEqual(result["status"], "ok")

    def test_protocol_rejects_multiple_or_truncated_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _parse_action(
                "<execute>result = 1</execute><execute>result = 2</execute>"
            )
        with self.assertRaisesRegex(ValueError, "truncated"):
            _parse_action(
                "<execute>result = 1</execute>", finish_reason="length"
            )
        self.assertEqual(
            _parse_action("analysis\n<execute>result = 1</execute>"),
            ("execute", "result = 1"),
        )
        self.assertEqual(
            _parse_action("analysis\n<execute>result = 1", finish_reason="stop"),
            ("execute", "result = 1"),
        )

    def test_caught_tool_failure_is_a_partial_observation(self) -> None:
        seen_messages: list[list[dict[str, str]]] = []
        replies = iter([
            "<execute>result = {'usable': 1}</execute>",
            "<solution>已根据部分观察完成。</solution>",
        ])

        def complete(_provider, _model, messages):
            seen_messages.append(list(messages))
            return {"choices": [{"message": {"content": next(replies)}}]}

        executor = _FixedExecution({
            "status": "ok",
            "result": {"usable": 1},
            "calls": [{
                "api": "cascast.forecast",
                "status": "error",
                "error": {"type": "RuntimeError", "message": "cache_only_miss"},
            }],
        })
        result = run_unified_agent(
            {}, "fake", "比较历史", {}, complete,
            data_descs=[], selected_tools=["ds"], executor=executor,
            max_steps=3,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["observations"][0]["status"], "partial")
        self.assertEqual(result["observations"][0]["tool_call_summary"]["failed"], 1)
        self.assertIn("cache_only_miss", seen_messages[1][-1]["content"])

    def test_execution_budget_preserves_a_solution_turn(self) -> None:
        replies = iter([
            "<execute>result = {'first': 1}</execute>",
            "<execute>result = {'second': 2}</execute>",
            "<solution>使用第一步观察完成回答。</solution>",
        ])
        executor = _FixedExecution({"status": "ok", "result": {"first": 1}, "calls": []})
        result = run_unified_agent(
            {}, "fake", "分析", {},
            lambda *_args: {"choices": [{"message": {"content": next(replies)}}]},
            data_descs=[], selected_tools=["ds"], executor=executor,
            max_steps=3, max_executions=1,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result["code_execution_attempt_count"], 1)
        self.assertTrue(any(
            item["action"] == "execution_budget" for item in result["agent_trace"]
        ))

    def test_preflight_rejection_does_not_consume_worker_execution_budget(self) -> None:
        replies = iter([
            "<execute>invalid generated code</execute>",
            "<solution>已根据预检错误停止。</solution>",
        ])
        executor = _FixedExecution({
            "status": "error",
            "error_type": "UnsafeCodeError",
            "error": "invalid Python syntax",
            "preflight": {"execution_started": False},
            "calls": [],
        })
        result = run_unified_agent(
            {}, "fake", "分析", {},
            lambda *_args: {"choices": [{"message": {"content": next(replies)}}]},
            data_descs=[], selected_tools=["ds"], executor=executor,
            max_steps=2, max_executions=1,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["code_execution_attempt_count"], 0)

    def test_multi_step_solution_gets_one_reflective_audit(self) -> None:
        seen_messages: list[list[dict[str, str]]] = []
        replies = iter([
            "<execute>result = {'first': 1}</execute>",
            "<execute>result = {'second': 2}</execute>",
            "<execute>result = {'third': 3}</execute>",
            "<solution>草稿结论。</solution>",
            "<solution>核对原问题与证据后的结论。</solution>",
        ])

        def complete(_provider, _model, messages):
            seen_messages.append(list(messages))
            return {"choices": [{"message": {"content": next(replies)}}]}

        result = run_unified_agent(
            {}, "fake", "复杂比较", {}, complete,
            data_descs=[], selected_tools=["ds"],
            executor=_FixedExecution({"status": "ok", "result": {"value": 1}, "calls": []}),
            max_steps=5,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "核对原问题与证据后的结论。")
        self.assertEqual(result["model_calls"], 5)
        self.assertTrue(any(
            item["action"] == "solution_review" for item in result["agent_trace"]
        ))
        self.assertIn("exact user question", seen_messages[4][-1]["content"])

    def test_old_observations_fold_while_newest_two_stay_verbatim(self) -> None:
        seen_messages: list[list[dict[str, str]]] = []
        replies = iter([
            "<execute>result = {'first': 1}</execute>",
            "<execute>result = {'second': 2}</execute>",
            "<execute>result = {'third': 3}</execute>",
            "<solution>草稿结论。</solution>",
            "<solution>最终结论。</solution>",
        ])

        def complete(_provider, _model, messages):
            seen_messages.append(list(messages))
            return {"choices": [{"message": {"content": next(replies)}}]}

        result = run_unified_agent(
            {}, "fake", "多步分析", {}, complete,
            data_descs=[], selected_tools=["ds"],
            executor=_FixedExecution({"status": "ok", "result": {"value": 1}, "calls": []}),
            max_steps=5,
        )
        self.assertEqual(result["status"], "ok")
        final_contents = [
            str(item["content"])
            for item in seen_messages[-1]
            if item["role"] == "user"
        ]
        folded = [
            text for text in final_contents
            if text.startswith("<observation-summary>")
        ]
        verbatim = [
            text for text in final_contents
            if text.startswith("<observation>\n")
        ]
        self.assertEqual(len(folded), 1)
        self.assertEqual(len(verbatim), 2)
        self.assertIn("Worker 命名空间", folded[0])

    def test_failed_execution_keeps_stdout_for_reflection(self) -> None:
        seen_messages: list[list[dict[str, str]]] = []
        replies = iter([
            "<execute>print('computed movement=2.5'); missing_name</execute>",
            "<solution>根据已计算的移动量完成回答。</solution>",
        ])

        def complete(_provider, _model, messages):
            seen_messages.append(list(messages))
            return {"choices": [{"message": {"content": next(replies)}}]}

        executor = _FixedExecution({
            "status": "error",
            "stdout": "computed movement=2.5\n",
            "error_type": "NameError",
            "error": "missing_name is not defined",
            "namespace_state": {
                "transaction": "rolled_back",
                "safe_to_reuse": ["earlier_value"],
                "uncommitted_names": ["computed"],
                "not_reached_assignment_names": ["result"],
            },
            "calls": [],
        })
        result = run_unified_agent(
            {}, "fake", "分析移动", {}, complete,
            data_descs=[], selected_tools=["ds"], executor=executor,
            max_steps=2, max_error_retries=3,
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("computed movement=2.5", seen_messages[1][-1]["content"])
        self.assertIn('"safe_to_reuse": ["earlier_value"]', seen_messages[1][-1]["content"])

        real_executor = ExecutionClient(timeout_seconds=10)
        try:
            failed = real_executor.execute(
                "retained_value = 2.5; print('retained-before-error'); missing_name",
                {}, data_descs=[], selected_tools=["ds"],
            )
        finally:
            real_executor.close()
        self.assertEqual(failed["status"], "error")
        self.assertIn("retained-before-error", failed["stdout"])
        self.assertNotIn("retained_value", failed["available_names"])
        self.assertEqual(failed["namespace_state"]["transaction"], "rolled_back")
        self.assertIn("retained_value", failed["namespace_state"]["uncommitted_names"])

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
            "os.environ", {"SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS": "90"}
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
        self.assertEqual(result["request_budget"]["limit_seconds"], 90.0)
        self.assertEqual(len(observed_timeouts), 1)
        self.assertGreater(observed_timeouts[0], 0)
        self.assertLessEqual(observed_timeouts[0], 0.5)

        with patch.dict(
            "os.environ",
            {
                "SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS": "150",
                "SEVIR_AGENT_MAX_REQUEST_SECONDS": "150",
            },
        ):
            self.assertEqual(request_budget_seconds(), 90.0)

        def late_complete(_provider, _model, _messages, _timeout_seconds):
            time.sleep(0.06)
            return {"choices": [{"message": {"content": "太晚了。"}}]}

        started = time.perf_counter()
        late_result = execute_unified_request(
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
        self.assertEqual(late_result["status"], "error")
        self.assertEqual(late_result["termination_reason"], "model_transport_failure")
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
            "requested_time_utc": "2019-06-27T03:00:00Z",
            "aurora_initialization_time_utc": "2019-06-27T03:00:00Z",
            "region": {"south": 41.3, "west": -111.98, "north": 53.05, "east": -88.23},
            "forecast_leads_hours": [6, 12, 18, 24],
        }
        descriptions, _ = bind_request_assets("分析R19062708588472", payload)
        replies = iter([
            "<expand>aurora</expand>",
            """<execute>
environment = aurora.regional_fields()
result = {
    'status': 'ok',
    'forecast_id': environment.attrs['forecast_id'],
    'cache_hit': environment.attrs['cache_hit'],
    'leads': environment.coords['lead_hour'].values.tolist(),
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
            max_steps=4,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["termination_reason"], "agent_completed")
        self.assertEqual(result["observations"][0]["status"], "ok")
        self.assertTrue(result["observations"][0]["result"]["cache_hit"])
        self.assertEqual(result["observations"][0]["result"]["leads"], [0, 6, 12, 18, 24])
        self.assertEqual(
            [item["action"] for item in result["agent_trace"]],
            ["expand", "execute", "finish"],
        )


if __name__ == "__main__":
    unittest.main()
