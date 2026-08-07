from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.agent.direct import run_direct_agent
from sevir_world.agent.reflective import run_reflective_agent
from sevir_world.agent.runtime import TOOL_CLASSES
from sevir_world.agent.tool_docs import extract_function_docstrings, selected_api_reference
from sevir_world.code_execution.client import SafeExecutionClient
from sevir_world.code_execution.safety import UnsafeCodeError, validate_code
from sevir_world.tools.geolocation import SevirGeolocator
from sevir_world.tools.metrics import _contingency, _fss_terms, _scores
from sevir_world.tools.uncertainty import track_fields


class SafetyValidatorTest(unittest.TestCase):
    def test_rejects_import_file_network_and_private_access(self) -> None:
        programs = (
            "import os\ndef run(ds, statistics):\n    return {}",
            "def run(ds, statistics):\n    return open('secret.txt').read()",
            "def run(ds, statistics):\n    return statistics.__dict__",
            "def run(ds, statistics):\n    return ds.__class__",
        )
        for program in programs:
            with self.subTest(program=program):
                with self.assertRaises(UnsafeCodeError):
                    validate_code(program, ("ds", "statistics"))

    def test_requires_exact_run_signature_tool_call_and_return(self) -> None:
        with self.assertRaisesRegex(UnsafeCodeError, "exactly one top-level run"):
            validate_code("result = {'value': 1}", ("ds", "statistics"))
        with self.assertRaisesRegex(UnsafeCodeError, "exactly match"):
            validate_code("def run(ds):\n    return ds", ("ds", "statistics"))
        with self.assertRaisesRegex(UnsafeCodeError, "selected tool"):
            validate_code(
                "def run(ds, statistics):\n    return {'count': len(ds)}",
                ("ds", "statistics"),
            )

    def test_dynamic_docs_include_only_selected_tools(self) -> None:
        documentation = selected_api_reference(("ds", "statistics"), TOOL_CLASSES)
        self.assertIn("statistics.compare_event", documentation)
        self.assertNotIn("forecaster.hazards", documentation)
        self.assertNotIn("world.", documentation)
        module = __import__(TOOL_CLASSES["statistics"].__module__, fromlist=["x"])
        _, methods = extract_function_docstrings(module.__file__, "SevirStatistics")
        self.assertIn("compare_event", {method["name"] for method in methods})


class WeatherAnalysisTest(unittest.TestCase):
    def test_offline_reverse_geocode_resolves_state_and_county(self) -> None:
        calls: list[dict] = []
        locator = SevirGeolocator(lambda api, arguments, result: calls.append(result))

        result = locator.reverse_geocode(39.482682, -90.151323)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["state_code"], "IL")
        self.assertEqual(result["county_fips"], "17117")

    def test_tracking_uses_mask_iou_and_bridges_one_missing_frame(self) -> None:
        fields = np.zeros((4, 20, 20), dtype=np.float32)
        fields[0, 5:9, 5:9] = 150
        fields[1, 5:9, 6:10] = 160
        fields[3, 5:9, 8:12] = 170

        result = track_fields(fields, threshold=133, minimum_region_pixels=4)

        self.assertEqual(result["track_count"], 1)
        self.assertEqual(result["tracks"][0]["maximum_gap_frames"], 1)
        self.assertGreater(result["tracks"][0]["path_length_km_approx"], 0)
        self.assertIn("connected-mask IoU", result["association_method"])

    def test_csi_hss_and_fss_are_computed_from_valid_pixels(self) -> None:
        prediction = np.array([[[2, 2], [0, 0]]], dtype=np.float32)
        target = np.array([[[2, 0], [2, 255]]], dtype=np.float32)
        counts = _contingency(prediction, target, threshold=1)
        scores = _scores(counts)
        numerator, denominator = _fss_terms(prediction, target, threshold=1, scale=1)

        self.assertEqual(counts["hits"], 1)
        self.assertEqual(counts["misses"], 1)
        self.assertEqual(counts["false_alarms"], 1)
        self.assertEqual(scores["csi"], round(1 / 3, 6))
        self.assertLess(scores["hss"], 0)
        self.assertEqual(round(1 - numerator / denominator, 6), 0.5)


class SafeExecutionIntegrationTest(unittest.TestCase):
    def test_executes_selected_tools_and_records_provenance(self) -> None:
        code = """
def run(ds, statistics):
    stats = statistics.historical()
    return {
        'loaded_datasets': len(ds),
        'catalog_records': stats.get('record_count', 0),
    }
"""
        with tempfile.TemporaryDirectory() as directory, patch(
            "sevir_world.code_execution.client.EXECUTION_LOG_ROOT", Path(directory)
        ):
            execution = SafeExecutionClient(timeout_seconds=30).execute(
                code, data_descs=[], selected_tools=["ds", "statistics"]
            )
            audit_path = Path(directory) / f"{execution['execution_id']}.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(execution["status"], "ok", execution)
        self.assertEqual(execution["result"]["loaded_datasets"], 0)
        self.assertGreater(execution["result"]["catalog_records"], 0)
        self.assertEqual(
            [call["api"] for call in execution["calls"]], ["statistics.historical"],
        )
        self.assertEqual(execution["limits"]["imports"], "disabled")
        self.assertEqual(audit["code"].strip(), code.strip())
        self.assertEqual(audit["selected_tools"], ["ds", "statistics"])
        self.assertEqual(audit["execution"]["status"], "ok")

    def test_data_indexer_loads_ds_before_run_and_tools_accept_it(self) -> None:
        code = """def run(ds, geolocator):
    location = geolocator.locate(ds)
    return {'event_id': ds[0].get('event_id'), 'center': location.get('center')}
"""
        with tempfile.TemporaryDirectory() as directory, patch(
            "sevir_world.code_execution.client.EXECUTION_LOG_ROOT", Path(directory)
        ):
            execution = SafeExecutionClient(timeout_seconds=30).execute(
                code,
                data_descs=[{"source": "sevir", "event_id": "R19100105107994"}],
                selected_tools=["ds", "geolocator"],
            )
        self.assertEqual(execution["status"], "ok", execution)
        self.assertEqual(execution["result"]["event_id"], "R19100105107994")
        self.assertEqual(execution["calls"][0]["api"], "geolocator.locate")


class _RepairingExecutor:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def execute(self, code: str, context: dict, **request: object) -> dict:
        self.codes.append(code)
        if len(self.codes) == 1:
            return {"status": "error", "error": "bad argument", "elapsed_ms": 1}
        return {
            "status": "ok",
            "result": {"combined": True, "value": 42},
            "primary_result": {"status": "ok", "task": "event_history_comparison"},
            "calls": [
                {
                    "step": 1,
                    "api": "data.event",
                    "arguments": {"event_id": context["event_id"]},
                    "result": {"status": "ok"},
                },
                {
                    "step": 2,
                    "api": "statistics.compare_event",
                    "arguments": {"event_id": context["event_id"]},
                    "result": {"status": "ok", "task": "event_history_comparison"},
                },
            ],
            "elapsed_ms": 2,
            "limits": {"imports": "disabled"},
        }


class DirectAgentTest(unittest.TestCase):
    def test_repairs_failed_code_and_synthesizes_from_execution(self) -> None:
        executor = _RepairingExecutor()
        replies = iter(
            (
                {"plan": "first", "code": "result = world.data.event()"},
                {
                    "plan": "combine metadata and history",
                    "code": (
                        "def run(ds, geolocator, forecaster, statistics, uncertainty, metrics, visualizer):\n"
                        "    return statistics.compare_event(ds[0].get('event_id'))"
                    ),
                },
            )
        )
        synthesis_messages: list[dict] = []

        def complete(provider: dict, model: str, messages: list[dict]) -> dict:
            if messages[0]["content"].startswith("Summarize"):
                synthesis_messages.extend(messages)
                return {"choices": [{"message": {"content": "执行结果显示比较已完成。"}}]}
            return {
                "choices": [
                    {"message": {"content": json.dumps(next(replies), ensure_ascii=False)}}
                ]
            }

        result = run_direct_agent(
            {"api_key": "test", "label": "Qwen"},
            "test-model",
            "比较当前事件和同类历史事件",
            {"event_id": "S123"},
            complete,
            executor=executor,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["direct_trace"]), 2)
        self.assertEqual(result["direct_trace"][0]["status"], "error")
        self.assertEqual(result["direct_trace"][1]["status"], "ok")
        self.assertEqual(len(result["tool_runs"]), 2)
        self.assertEqual(result["code_result"]["value"], 42)
        self.assertIn('"value": 42', synthesis_messages[1]["content"])


class _ReflectiveExecutor:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def execute(self, code: str, context: dict, **request: object) -> dict:
        self.codes.append(code)
        if len(self.codes) == 1:
            result = {"event_id": context["event_id"], "maximum_vil": 180}
            api = "forecaster.nowcast"
        else:
            result = {"hazard_regions": 2, "duration_minutes": 35}
            api = "forecaster.hazards"
        return {
            "status": "ok",
            "result": result,
            "primary_result": result,
            "calls": [
                {
                    "step": 1,
                    "api": api,
                    "arguments": {"event_id": context["event_id"]},
                    "result": result,
                }
            ],
            "elapsed_ms": 1,
            "execution_id": f"execution-{len(self.codes)}",
            "limits": {"imports": "disabled"},
        }


class ReflectiveAgentTest(unittest.TestCase):
    def test_executes_observes_refines_then_finishes(self) -> None:
        executor = _ReflectiveExecutor()
        replies = iter(
            (
                {
                    "action": "execute",
                    "plan": "先查看未来回波",
                    "code": (
                        "def run(ds, geolocator, forecaster, statistics, uncertainty, metrics, visualizer):\n"
                        "    return forecaster.nowcast(ds)"
                    ),
                },
                {
                    "action": "execute",
                    "plan": "根据回波进一步检查危险区域",
                    "code": (
                        "def run(ds, geolocator, forecaster, statistics, uncertainty, metrics, visualizer):\n"
                        "    return forecaster.hazards(ds, threshold=133)"
                    ),
                },
                {
                    "action": "finish",
                    "answer": "未来存在2个较强回波候选区域，窗口内持续35分钟。",
                },
            )
        )
        seen_messages: list[list[dict]] = []

        def complete(provider: dict, model: str, messages: list[dict]) -> dict:
            seen_messages.append([dict(item) for item in messages])
            return {
                "choices": [
                    {"message": {"content": json.dumps(next(replies), ensure_ascii=False)}}
                ]
            }

        result = run_reflective_agent(
            {"api_key": "test", "label": "Qwen"},
            "test-model",
            "预测并分析危险区域和持续时间",
            {"event_id": "S123"},
            complete,
            executor=executor,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["route_mode"], "zephyrus_reflective")
        self.assertEqual(len(executor.codes), 2)
        self.assertEqual(len(result["observations"]), 2)
        self.assertEqual(len(result["tool_runs"]), 2)
        self.assertEqual(result["reflective_trace"][-1]["action"], "finish")
        self.assertIn("2个", result["answer"])
        self.assertIn("<observation>", seen_messages[1][-1]["content"])
        self.assertIn("maximum_vil", seen_messages[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
