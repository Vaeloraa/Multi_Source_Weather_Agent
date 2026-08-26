from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sevir_world.agent.runtime import ToolCostBudgetExceeded, ToolUsageRecorder
from sevir_world.agent.tool_docs import capability_index, selected_api_reference
from sevir_world.agent.unified import _capabilities_used_in_code, run_unified_agent
from sevir_world.api.response_pipeline import finalize_agent_response
from sevir_world.capabilities.base import BoundCapability, RequestBinding
import sevir_world
import xarray as xr
from sevir_world.tools.sevir.Sevir import Sevir
from sevir_world.code_execution.worker import _json_safe


class ExecutionProtocolArchitectureTests(unittest.TestCase):
    def test_cancelled_request_still_persists_diagnostic_trace(self) -> None:
        class Trace:
            def record_stage(self, *args, **kwargs):
                pass

            def add(self, **kwargs):
                pass

            def finish(self, status):
                return {"status": status, "failure_class": "client_disconnected"}

        started = time.perf_counter()
        with patch.object(Path, "write_text") as write_text:
            result = finalize_agent_response(
                {"status": "error", "termination_reason": "client_cancelled"},
                payload={"trace": True, "session_id": "cancelled-trace"},
                question="timeout diagnosis",
                request_id="request-timeout",
                mode="unified",
                provider_name="fake",
                model_name="fake",
                route_decision={},
                started=started,
                analysis_started=started,
                trace=Trace(),
                presentation={},
                artifact_root=Path("tests/_trace_output"),
                cancelled=lambda: True,
            )

        self.assertEqual(result["finalization_deferred"], "client_cancelled")
        persisted = json.loads(write_text.call_args.args[0])
        self.assertEqual(persisted["finalization_deferred"], "client_cancelled")
        self.assertEqual(
            persisted["request_trace"]["failure_class"], "client_disconnected"
        )

    def test_scientific_transport_marks_lossy_summaries_explicitly(self) -> None:
        frame_summary = _json_safe(pd.DataFrame({"event_id": ["R1"]}))
        array_summary = _json_safe(xr.DataArray(np.zeros((2, 2)), dims=("row", "column")))
        dataset_summary = _json_safe(xr.Dataset({"vil": (("row", "column"), np.zeros((2, 2)))}))
        for summary in (frame_summary, array_summary, dataset_summary):
            self.assertEqual(summary["representation"], "metadata_only")
            self.assertEqual(summary["read_scope"], "worker_only")

    def test_runtime_cost_is_reserved_before_backend_and_persists_on_cache_hit(self) -> None:
        recorder = ToolUsageRecorder(1.25)
        capability = BoundCapability(recorder, RequestBinding())
        backend_calls = 0

        def backend() -> dict[str, str]:
            nonlocal backend_calls
            backend_calls += 1
            return {"status": "ok", "value": "evidence"}

        first = capability._invoke("sevir.event", {}, backend, lambda raw: raw["value"])
        second = capability._invoke("sevir.event", {}, backend, lambda raw: raw["value"])
        self.assertEqual((first, second), ("evidence", "evidence"))
        self.assertEqual(backend_calls, 1)
        self.assertEqual(recorder.summary()["used_units"], 1.25)
        with self.assertRaises(ToolCostBudgetExceeded):
            capability._invoke("sevir.event", {"different": True}, backend, lambda raw: raw)
        self.assertEqual(backend_calls, 1)

    def test_batch_observations_is_one_public_call_and_returns_plain_value(self) -> None:
        recorder = ToolUsageRecorder(20)
        tool = Sevir(recorder, RequestBinding())

        def record(event_id: str) -> dict[str, object]:
            return {
                "status": "ok",
                "event_id": event_id,
                "subject_key": f"sevir:{event_id.lower()}",
                "event_anchor_time_utc": "2019-06-27T00:00:00Z",
                "frame_times_utc": ["2019-06-27T00:00:00Z"],
                "values": np.zeros((1, 2, 2), dtype=np.float32),
                "relative_minutes": [0],
                "missing_value": 255,
            }

        backend = {
            "status": "partial",
            "records": [record("R1"), record("R2")],
            "unavailable": [{"subject_key": "sevir:r3", "reason": "missing"}],
        }
        with patch(
            "sevir_world.tools.sevir.Catalog.load_sevir_vil_events",
            return_value=backend,
        ):
            result = tool.observation_batch(["sevir:r1", "sevir:r2", "sevir:r3"])
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(result.attrs["loaded_event_ids"], ("R1", "R2"))
        self.assertEqual(result.attrs["unavailable_subject_keys"], ("sevir:r3",))
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0]["api"], "sevir.observation_batch")
        self.assertEqual(
            recorder.calls[0]["arguments"]["subject_keys"],
            ("sevir:r1", "sevir:r2", "sevir:r3"),
        )
        self.assertNotIn("event_ids", recorder.calls[0]["arguments"])

    def test_batch_observations_outer_aligns_different_frame_counts(self) -> None:
        recorder = ToolUsageRecorder(20)
        tool = Sevir(recorder, RequestBinding())

        def record(event_id: str, offsets: list[int]) -> dict[str, object]:
            return {
                "status": "ok",
                "event_id": event_id,
                "subject_key": f"sevir:{event_id.lower()}",
                "event_anchor_time_utc": "2019-06-27T00:00:00Z",
                "frame_times_utc": [
                    f"2019-06-27T00:{index:02d}:00Z"
                    for index in range(len(offsets))
                ],
                "values": np.ones((len(offsets), 2, 2), dtype=np.float32),
                "relative_minutes": offsets,
                "missing_value": 255,
            }

        backend = {
            "status": "ok",
            "records": [
                record("R12", list(range(-55, 5, 5))),
                record("R13", list(range(-60, 5, 5))),
            ],
            "unavailable": [],
        }
        with patch(
            "sevir_world.tools.sevir.Catalog.load_sevir_vil_events",
            return_value=backend,
        ):
            result = tool.observation_batch(["sevir:r12", "sevir:r13"])

        self.assertEqual(result["vil"].dims, ("event", "time", "row", "column"))
        self.assertEqual(result.sizes["time"], 13)
        self.assertEqual(result.coords["time"].values.tolist(), list(range(-60, 5, 5)))
        self.assertTrue(np.isnan(result["vil"].isel(event=0, time=0)).all())

    def test_capability_contract_is_layered_and_folded_after_execute(self) -> None:
        index = capability_index(("ds", "sevir", "geolocator", "cascast", "aurora"))
        detail = selected_api_reference(("sevir",))
        self.assertLess(len(index), len(detail))
        self.assertIn("<expand>capability_name</expand>", index)

        replies = iter((
            "<expand>sevir, geolocator</expand>",
            "<execute>result = {'stage': 'postprocess'}</execute>",
            "<solution>已依据现有证据回答。</solution>",
        ))
        prompts: list[list[dict[str, str]]] = []

        def complete(provider, model, messages):
            prompts.append([dict(item) for item in messages])
            return {
                "choices": [{
                    "message": {"content": next(replies)},
                    "finish_reason": "stop",
                }]
            }

        class Executor:
            def execute(self, *args, **kwargs):
                return {
                    "status": "error",
                    "error_type": "ZeroDivisionError",
                    "error": "postprocessing failed",
                    "calls": [{
                        "api": "sevir.event",
                        "arguments": {},
                        "status": "ok",
                        "return_type": "Event",
                        "result": {"type": "Event", "id": "R1"},
                        "cost_units": 1.0,
                    }],
                    "tool_cost_summary": {
                        "budget_units": 30.0,
                        "used_units": 1.0,
                        "remaining_units": 29.0,
                        "exhausted": False,
                    },
                    "evidence_acquired": True,
                    "evidence_tool_call_count": 1,
                    "postprocessing_succeeded": False,
                }

        result = run_unified_agent(
            {}, "fixed", "测试分层契约", {}, complete,
            executor=Executor(), max_steps=3,
        )
        self.assertEqual(result["evidence_execution_count"], 1)
        self.assertEqual(result["evidence_tool_call_count"], 1)
        self.assertEqual(result["postprocessing_success_count"], 0)
        second_prompt = "\n".join(item["content"] for item in prompts[1])
        self.assertIn('<capability-contracts names="sevir">', second_prompt)
        self.assertNotIn('<capability-contracts names="sevir,geolocator">', second_prompt)
        self.assertIn("other requested capability contracts were deferred", second_prompt)
        third_prompt = "\n".join(item["content"] for item in prompts[2])
        self.assertIn("<capability-contract-summary", third_prompt)
        self.assertNotIn("<capability-contracts names=", third_prompt)
        self.assertIn(
            "sevir.observations(subject_key: str | None = None",
            third_prompt,
        )
        self.assertIn("dimensions ``time``, ``row`` and ``column``", third_prompt)
        self.assertIn('<repair-contracts names="sevir">', third_prompt)
        self.assertNotIn("event_ids: list", third_prompt)

    def test_capability_must_be_revealed_before_generated_code_can_use_it(self) -> None:
        self.assertEqual(
            _capabilities_used_in_code("event = sevir.event(); result = event.id"),
            {"sevir"},
        )
        replies = iter((
            "<execute>event = sevir.event(); result = event.id</execute>",
            "<expand>sevir</expand>",
            "<execute>event = sevir.event(); result = event.id</execute>",
            "<solution>完成。</solution>",
        ))

        def complete(provider, model, messages):
            return {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}]}

        class Executor:
            calls = 0

            def execute(self, *args, **kwargs):
                self.calls += 1
                return {
                    "status": "ok",
                    "result": {"event_id": "R1"},
                    "calls": [],
                    "generated_artifacts": [],
                }

        executor = Executor()
        result = run_unified_agent(
            {}, "fixed", "读取事件", {}, complete,
            executor=executor, max_steps=4,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result["code_execution_attempt_count"], 1)
        self.assertEqual(result["agent_trace"][0]["action"], "contract_required")

    def test_public_time_fields_do_not_alias_legacy_analysis_time(self) -> None:
        legacy = RequestBinding.from_mapping({
            "analysis_time_utc": "2019-06-27T00:00:00Z",
        })
        self.assertEqual(legacy.requested_time_utc, "")
        self.assertEqual(legacy.aurora_initialization_time_utc, "")
        explicit = RequestBinding.from_mapping({
            "requested_time_utc": "2019-06-27T01:00:00Z",
            "aurora_initialization_time_utc": "2019-06-27T00:00:00Z",
        })
        self.assertNotEqual(
            explicit.requested_time_utc,
            explicit.aurora_initialization_time_utc,
        )


if __name__ == "__main__":
    unittest.main()
