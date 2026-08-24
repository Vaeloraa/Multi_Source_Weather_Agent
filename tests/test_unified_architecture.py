from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sevir_world.agent.unified import run_unified_agent
from sevir_world.api.model_client import (
    FIXED_MODEL_NAME,
    PROVIDERS,
    fixed_model_selection,
    post_chat_completion,
)
from sevir_world.api.unified_service import bind_request_assets
from sevir_world.code_execution.client import ExecutionClient


ROOT = Path(__file__).resolve().parents[1]


class UnifiedArchitectureTests(unittest.TestCase):
    def test_runtime_contains_only_one_agent_path(self) -> None:
        server = (ROOT / "sevir_world" / "api" / "server.py").read_text(encoding="utf-8")
        agent_files = {
            path.name for path in (ROOT / "sevir_world" / "agent").glob("*.py")
        }
        self.assertNotIn("direct.py", agent_files)
        self.assertNotIn("reflective.py", agent_files)
        self.assertNotIn("task_planning.py", agent_files)
        self.assertNotIn("claim_gate.py", agent_files)
        self.assertNotIn("SYSTEM_PROMPT", server)
        self.assertIn("execute_unified_request", server)

    def test_event_id_adjacent_to_chinese_text_binds_real_assets(self) -> None:
        payload: dict[str, object] = {
            "mode": "reflective",
            "task_spec": {"old": True},
        }
        descriptions, tools = bind_request_assets(
            "请问R19062708588472未来多路预测分歧大不大", payload
        )
        self.assertEqual(payload["event_id"], "R19062708588472")
        self.assertEqual(descriptions[0]["representation"], "event_handle")
        self.assertNotIn("observation_window_minutes", descriptions[0])
        self.assertEqual(tools, ["ds", "sevir", "geolocator", "cascast", "aurora"])
        self.assertNotIn("mode", payload)
        self.assertNotIn("task_spec", payload)

    def test_execute_steps_share_complete_python_namespace(self) -> None:
        client = ExecutionClient(timeout_seconds=20, memory_limit_mb=1024)
        try:
            first = client.execute(
                "import statistics as stats\n"
                "working_array = np.arange(6, dtype=np.float32)\n"
                "def retained_transform(offset):\n"
                "    return float(stats.mean(working_array)) + offset\n"
                "result = {'value': retained_transform(1)}",
                {},
                data_descs=[],
                selected_tools=("ds",),
            )
            second = client.execute(
                "result = {\n"
                "    'continued': retained_transform(2),\n"
                "    'dtype': str(working_array.dtype),\n"
                "    'module': stats.__name__,\n"
                "}",
                {},
                data_descs=[],
                selected_tools=("ds",),
            )
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["result"]["value"], 3.5)
            self.assertEqual(second["result"]["continued"], 4.5)
            self.assertEqual(second["result"]["dtype"], "float32")
            self.assertEqual(second["result"]["module"], "statistics")
            self.assertEqual(
                first["worker_session"]["id"], second["worker_session"]["id"]
            )
            self.assertEqual(second["worker_session"]["step"], 2)
        finally:
            client.close()

    def test_lazy_capability_exposes_documented_signature_without_backend_import(self) -> None:
        result = ExecutionClient(timeout_seconds=20, memory_limit_mb=1024).execute(
            """
import inspect
import sys
result = {
    'signature': str(inspect.signature(cascast.forecast)),
    'capability_loaded': 'sevir_world.tools.capabilities' in sys.modules,
    'forecast_backend_loaded': 'sevir_world.backends.forecast' in sys.modules,
}
""",
            {},
            data_descs=[],
            selected_tools=("ds", "cascast"),
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("event_id", result["result"]["signature"])
        self.assertTrue(result["result"]["capability_loaded"])
        self.assertFalse(result["result"]["forecast_backend_loaded"])

    def test_all_capabilities_remain_unloaded_until_first_access(self) -> None:
        result = ExecutionClient(timeout_seconds=20, memory_limit_mb=1024).execute(
            """
import sys
result = {
    'capability_module': 'sevir_world.tools.capabilities' in sys.modules,
    'forecast_backend': 'sevir_world.backends.forecast' in sys.modules,
    'geospatial_backend': 'sevir_world.backends.geospatial' in sys.modules,
    'foundation_backend': 'sevir_world.tools.live_foundation' in sys.modules,
}
""",
            {},
            data_descs=[],
            selected_tools=("ds", "sevir", "geolocator", "cascast", "aurora"),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"], {
            "capability_module": False,
            "forecast_backend": False,
            "geospatial_backend": False,
            "foundation_backend": False,
        })

    def test_unselected_heavy_tools_are_not_imported(self) -> None:
        client = ExecutionClient(timeout_seconds=20, memory_limit_mb=1024)
        try:
            result = client.execute(
                "import sys\n"
                "result = {\n"
                "    'visualization': 'sevir_world.tools.visualization' in sys.modules,\n"
                "    'forecast': 'sevir_world.tools.forecast' in sys.modules,\n"
                "    'matplotlib': 'matplotlib' in sys.modules,\n"
                "    'torch': 'torch' in sys.modules,\n"
                "}",
                {},
                data_descs=[],
                selected_tools=("ds",),
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(
                result["result"],
                {
                    "visualization": False,
                    "forecast": False,
                    "matplotlib": False,
                    "torch": False,
                },
            )
            self.assertLess(result["elapsed_ms"], 10_000)
        finally:
            client.close()

    def test_plain_natural_language_is_a_terminal_answer(self) -> None:
        calls = 0

        def complete(_provider, _model, _messages):
            nonlocal calls
            calls += 1
            return {"choices": [{"message": {"content": "这是一个完整回答。"}}]}

        result = run_unified_agent(
            {}, "fake", "这个系统能做什么？", {}, complete,
            data_descs=[], selected_tools=["ds"],
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "这是一个完整回答。")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["code_execution_count"], 0)
        self.assertEqual(calls, 1)

    def test_request_bound_model_and_event_are_visible_to_agent(self) -> None:
        prompts: list[str] = []

        def complete(_provider, _model, messages):
            prompts.append(messages[-1]["content"])
            return {"choices": [{"message": {"content": "已完成。"}}]}

        result = run_unified_agent(
            {}, "fake", "分析这个事件", {
                "event_id": "R19062708588472",
                "data_policy": "cached",
                "weather_model": "cascast_live",
                "_deadline_monotonic": 123.0,
            }, complete, data_descs=[], selected_tools=["ds"],
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn('"event_id": "R19062708588472"', prompts[0])
        self.assertIn('"weather_model": "cascast_live"', prompts[0])
        self.assertNotIn("_deadline_monotonic", prompts[0])

    def test_r190_binds_only_a_lightweight_event_handle(self) -> None:
        payload: dict[str, object] = {
            "event_id": "R19062708588472",
            "data_policy": "cached",
            "weather_model": "cascast_live",
        }
        descriptions, _ = bind_request_assets("分析R19062708588472", payload)
        self.assertEqual(len(descriptions), 1)
        self.assertEqual(descriptions[0]["source"], "sevir")
        self.assertEqual(descriptions[0]["representation"], "event_handle")
        self.assertNotIn("foundation_cache_binding", payload)
        self.assertNotIn("analysis_time_utc", payload)

    def test_initial_prompt_contains_complete_small_tool_contract(self) -> None:
        prompts: list[str] = []

        def complete(_provider, _model, messages):
            prompts.append(messages[-1]["content"])
            return {"choices": [{"message": {"content": "已完成。"}}]}

        payload: dict[str, object] = {
            "event_id": "R19062708588472",
            "data_policy": "cached",
            "weather_model": "cascast_live",
        }
        descriptions, tools = bind_request_assets("分析R19062708588472", payload)
        result = run_unified_agent(
            {}, "fake", "分析R19062708588472", dict(payload), complete,
            data_descs=descriptions, selected_tools=tools,
        )
        self.assertEqual(result["status"], "ok")
        self.assertLess(len(prompts[0]), 12_000)
        self.assertIn("sevir.event(event_id: str | None = None)", prompts[0])
        self.assertIn("cascast.forecast(event_id: str | None = None", prompts[0])
        self.assertIn("aurora.regional_fields(analysis_time_utc: str", prompts[0])
        self.assertIn("6-24 hour lead times", prompts[0])
        self.assertNotIn("�", prompts[0])
        self.assertNotIn("dir(tool)", prompts[0])
        self.assertNotIn("inspect.signature", prompts[0])
        self.assertNotIn("foundation_live_cache", prompts[0])
        self.assertNotIn("384, 384", prompts[0])
        self.assertFalse(result["dataset_description"]["details_expanded"])

    def test_r190_foundation_cache_is_readable_inside_exec_namespace(self) -> None:
        payload: dict[str, object] = {
            "event_id": "R19062708588472",
            "data_policy": "cached",
            "weather_model": "cascast_live",
        }
        descriptions, _ = bind_request_assets("分析R19062708588472", payload)
        context = {
            key: payload.get(key)
            for key in (
                "event_id", "data_policy", "weather_model", "analysis_time_utc",
                "latitude", "longitude", "forecast_leads_hours",
                "history_before_hours", "history_after_hours",
            )
        }
        result = ExecutionClient(timeout_seconds=20, memory_limit_mb=2048).execute(
            """
case = aurora.case()
result = {
    'status': case.status,
    'cache_hit': case.cache.cache_hit,
    'leads': case.evidence.aurora.rollout_leads_hours,
    'verification': case.evidence.aurora.verification_status,
}
""",
            context,
            data_descs=descriptions,
            selected_tools=("ds", "aurora"),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["result"]["status"], "ok")
        self.assertTrue(result["result"]["cache_hit"])
        self.assertEqual(result["result"]["leads"], [6, 12, 18, 24])
        self.assertEqual(result["result"]["verification"], "full_domain_multi_lead")

    def test_runtime_registers_plain_png_names(self) -> None:
        client = ExecutionClient(timeout_seconds=30, memory_limit_mb=2048)
        result = client.execute(
            """
import os
import matplotlib.pyplot as plt
path = os.path.join(os.environ['SEVIR_ARTIFACT_DIR'], 'plain-test-name.png')
plt.plot([0, 1], [1, 0])
plt.savefig(path)
plt.close()
result = {'plot': path}
""",
            {},
            data_descs=[],
            selected_tools=("ds",),
        )
        self.assertEqual(result["status"], "ok")
        url = result["result"]["plot"]
        self.assertRegex(url, r"^/api/artifacts/[0-9a-f]{32}\.png$")
        self.assertEqual(len(result.get("generated_artifacts", [])), 1)

    def test_one_agent_can_execute_observe_continue_and_finish(self) -> None:
        replies = iter([
            "<execute>result = {'value': 20}</execute>",
            "<execute>result = {'value': result['value'] * 2}</execute>",
            "<solution>结果是 40。</solution>",
        ])

        def complete(_provider, _model, _messages):
            return {
                "choices": [{"message": {"content": next(replies)}}]
            }

        result = run_unified_agent(
            {"label": "fake", "api_key": "unused"},
            "fake",
            "计算结果",
            {},
            complete,
            data_descs=[],
            selected_tools=["ds"],
            max_steps=4,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "结果是 40。")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["code_execution_count"], 2)
        self.assertEqual(
            [item["action"] for item in result["agent_trace"]],
            ["execute", "execute", "finish"],
        )

    def test_execution_trace_contains_human_readable_tool_facts(self) -> None:
        replies = iter([
            "<execute>result = sevir.event('R19062708588472')</execute>",
            "<solution>已读取事件资料。</solution>",
        ])

        class StubExecutor:
            def execute(self, *_args, **_kwargs):
                return {
                    "status": "ok",
                    "result": {"status": "ok", "event_id": "R19062708588472"},
                    "calls": [{
                        "api": "sevir.event",
                        "arguments": {"event_id": "R19062708588472"},
                        "status": "ok",
                        "result_status": "ok",
                        "cache_hit": True,
                        "tool_contract": {"output_role": "event metadata"},
                        "result": {
                            "status": "ok",
                            "event_id": "R19062708588472",
                            "cache_hit": True,
                        },
                    }],
                    "generated_artifacts": [],
                }

        def complete(_provider, _model, _messages):
            return {"choices": [{"message": {"content": next(replies)}}]}

        result = run_unified_agent(
            {}, "fake", "读取这个事件", {}, complete,
            data_descs=[], selected_tools=["ds", "sevir"], executor=StubExecutor(),
        )
        call = result["agent_trace"][0]["calls"][0]
        self.assertEqual(call["purpose"], "event metadata")
        self.assertIn("事件 R19062708588472", call["result_summary"])
        self.assertIn("命中已有数据缓存", call["result_summary"])

    def test_fixed_model_failure_does_not_try_another_candidate(self) -> None:
        provider = PROVIDERS["qwen"]
        with patch(
            "sevir_world.api.model_client._call_once",
            side_effect=RuntimeError("Free quota exhausted"),
        ) as call:
            with self.assertRaisesRegex(RuntimeError, "未尝试其他模型"):
                post_chat_completion(
                    provider,
                    FIXED_MODEL_NAME,
                    [{"role": "user", "content": "OK"}],
                    timeout_seconds=10,
                )
        self.assertEqual(call.call_count, 1)

    def test_server_model_selection_is_fixed(self) -> None:
        provider_name, model = fixed_model_selection(PROVIDERS)
        self.assertEqual(provider_name, "qwen")
        self.assertEqual(model, "deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
