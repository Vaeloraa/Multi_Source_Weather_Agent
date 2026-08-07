from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.direct_agent import run_direct_agent
from sevir_world.execution import SafeExecutionClient
from sevir_world.geolocation import SevirGeolocator
from sevir_world.metrics import _contingency, _fss_terms, _scores
from sevir_world.safety import UnsafeCodeError, validate_code
from sevir_world.uncertainty import track_fields


class SafetyValidatorTest(unittest.TestCase):
    def test_rejects_import_file_network_and_private_access(self) -> None:
        programs = (
            "import os\nresult = world.data.uploads()",
            "result = open('secret.txt').read()\nworld.data.uploads()",
            "result = world.__dict__",
            "result = context.__class__\nworld.data.uploads()",
        )
        for program in programs:
            with self.subTest(program=program):
                with self.assertRaises(UnsafeCodeError):
                    validate_code(program)

    def test_requires_world_call_and_result_assignment(self) -> None:
        with self.assertRaisesRegex(UnsafeCodeError, "approved world API"):
            validate_code("result = {'value': 1}")
        with self.assertRaisesRegex(UnsafeCodeError, "assign.*result"):
            validate_code("events = world.data.search(limit=1)")


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
    def test_executes_multiple_world_tools_and_records_provenance(self) -> None:
        code = """
events = world.data.search(query='', limit=1)
stats = world.statistics.historical()
result = {
    'returned_events': len(events.get('records', [])),
    'catalog_records': stats.get('record_count', 0),
}
"""
        with tempfile.TemporaryDirectory() as directory, patch(
            "sevir_world.execution.EXECUTION_LOG_ROOT", Path(directory)
        ):
            execution = SafeExecutionClient(timeout_seconds=30).execute(code)
            audit_path = Path(directory) / f"{execution['execution_id']}.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(execution["status"], "ok", execution)
        self.assertEqual(execution["result"]["returned_events"], 1)
        self.assertGreater(execution["result"]["catalog_records"], 0)
        self.assertEqual(
            [call["api"] for call in execution["calls"]],
            ["data.search", "statistics.historical"],
        )
        self.assertEqual(execution["limits"]["imports"], "disabled")
        self.assertEqual(audit["code"].strip(), code.strip())
        self.assertEqual(audit["execution"]["status"], "ok")


class _RepairingExecutor:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def execute(self, code: str, context: dict) -> dict:
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
                        "event = world.data.event(context.get('event_id'))\n"
                        "result = world.statistics.compare_event(context.get('event_id'))"
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


if __name__ == "__main__":
    unittest.main()
