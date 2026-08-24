from __future__ import annotations

import unittest
from unittest.mock import patch

from sevir_world.tools.base import WorldComponent, _encode_request_cache
from sevir_world.code_execution.client import ExecutionClient


class RequestToolCacheTests(unittest.TestCase):
    def test_different_program_reuses_tool_result_after_postprocessing_failure(self) -> None:
        client = ExecutionClient(timeout_seconds=15)
        # Disable the optional filesystem cache: this test specifically proves
        # parent-host checkpoint replay across two different worker programs.
        client._request_tool_cache_dir = ""
        failed = client.execute(
            "def run(ds, sevir):\n"
            "    event = sevir.event('R19062708588472')\n"
            "    return missing_after_tool_call\n",
            {},
            data_descs=[],
            selected_tools=["ds", "sevir"],
        )
        repaired = client.execute(
            "def run(ds, sevir):\n"
            "    event = sevir.event('R19062708588472')\n"
            "    return {'event_id': event['record']['id']}\n",
            {},
            data_descs=[],
            selected_tools=["ds", "sevir"],
        )

        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["calls"][0]["status"], "ok")
        self.assertFalse(failed["calls"][0]["cache_hit"])
        self.assertEqual(repaired["status"], "ok")
        self.assertTrue(repaired["calls"][0]["cache_hit"])

    def test_small_result_is_reused_across_component_instances(self) -> None:
        backend_calls = 0
        recorded = []

        def backend(*, event_id: str):
            nonlocal backend_calls
            backend_calls += 1
            return {"status": "ok", "event_id": event_id, "value": 7}

        def recorder(name, arguments, result, error=None, cache_hit=False):
            recorded.append({"name": name, "cache_hit": cache_hit})

        shared_cache = {}
        with patch(
            "sevir_world.tools.base._request_cache_path", side_effect=lambda key: key
        ), patch(
            "sevir_world.tools.base._read_request_cache",
            side_effect=lambda key: shared_cache.get(key),
        ), patch(
            "sevir_world.tools.base._write_request_cache",
            side_effect=lambda key, value: shared_cache.__setitem__(key, value),
        ):
            first = WorldComponent(recorder)._call(
                "statistics.values", {"event_id": "R19062708588472"}, backend
            )
            second = WorldComponent(recorder)._call(
                "statistics.values", {"event_id": "R19062708588472"}, backend
            )

        self.assertEqual(first["value"], 7)
        self.assertEqual(second["value"], 7)
        self.assertEqual(backend_calls, 1)
        self.assertTrue(recorded[-1]["cache_hit"])


    def test_non_json_result_is_not_cacheable(self) -> None:
        self.assertIsNone(_encode_request_cache({"opaque": object()}))
        self.assertEqual(
            _encode_request_cache({"status": "ok", "value": 7}),
            '{"status": "ok", "value": 7}',
        )
