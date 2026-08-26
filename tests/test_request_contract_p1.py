from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest.mock import patch

from sevir_world.agent.contracts import RequestContext
from sevir_world.agent.tool_docs import selected_api_reference
from sevir_world.api.unified_service import _execute_unified_request_held
from sevir_world.capabilities.base import (
    ForecastSpecification,
    RegionBounds,
    RequestAssets,
    RequestBinding,
    RequestTarget,
)


class RequestContractP1Tests(unittest.TestCase):
    def test_binding_exposes_grouped_immutable_request_values(self) -> None:
        binding = RequestBinding.from_mapping({
            "event_id": "r1",
            "requested_time_utc": "2019-06-27 03:00:00",
            "latitude": 37.2,
            "longitude": -93.3,
            "region": {"south": 30, "west": -100, "north": 40, "east": -90},
            "forecast_leads": [6, 12],
            "history_before_hours": 0,
            "history_after_hours": 0,
            "data_policy": "LIVE",
        })
        self.assertEqual(binding.assets, RequestAssets("R1", ""))
        self.assertEqual(binding.target, RequestTarget(
            "2019-06-27T03:00:00Z",
            37.2,
            -93.3,
            RegionBounds(30, -100, 40, -90),
        ))
        self.assertEqual(binding.forecast, ForecastSpecification(
            initialization_time_utc="",
            region=RegionBounds(30, -100, 40, -90),
            lead_hours=(6, 12),
            history_before_hours=0,
            history_after_hours=0,
        ))
        self.assertEqual(binding.policy.mode, "live")

    def test_binding_round_trips_through_worker_json_shape(self) -> None:
        original = RequestBinding.from_mapping({
            "event_id": "R1",
            "region": {"south": 30, "west": -100, "north": 40, "east": -90},
            "event_resolution": {"source": "explicit_request"},
            "requested_time_source": "question",
        })
        self.assertEqual(RequestBinding.from_mapping(asdict(original)), original)

    def test_invalid_or_ambiguous_request_facts_fail_before_agent_execution(self) -> None:
        invalid = (
            {"requested_time_utc": "not-a-time"},
            {"latitude": 10},
            {"latitude": 91, "longitude": 0},
            {"region": {"south": 40, "west": -100, "north": 30, "east": -90}},
            {"forecast_leads_hours": []},
            {"history_before_hours": -1},
            {"data_policy": "whatever"},
            {"upload_id": "a", "sequence_upload_id": "b"},
            {"forecast_leads_hours": [6], "forecast_leads": [12]},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RequestBinding.from_mapping(value)

    def test_api_builds_one_execution_request_after_ingress_resolution(self) -> None:
        payload = {
            "session_id": "p1-single-request",
            "event_id": "R1",
            "event_resolution": {"source": "explicit_request"},
            "requested_time_utc": "2019-06-27T03:00:00Z",
            "forecast_leads": [3, 6],
            "data_descs": [{"source": "sevir", "event_id": "R1"}],
            "selected_tools": ["ds", "sevir"],
            "_cancelled": lambda: False,
        }
        with patch("sevir_world.api.unified_service.run_unified_agent", return_value={}) as run:
            _execute_unified_request_held(
                "qwen", "fixed", "test", payload,
                providers={"qwen": {}},
                complete=lambda *_args: {},
            )
        context = run.call_args.args[3]
        request = run.call_args.kwargs["execution_request"]
        self.assertIsInstance(request.context, RequestContext)
        self.assertEqual(set(context), {"_deadline_monotonic", "_cancelled"})
        self.assertEqual(request.binding.event_id, "R1")
        self.assertEqual(request.binding.forecast.lead_hours, (3, 6))
        self.assertEqual(request.binding.provenance.event, "explicit_request")
        self.assertEqual(request.data_descs[0]["event_id"], "R1")
        self.assertNotIn("data_descs", run.call_args.kwargs)
        self.assertNotIn("selected_tools", run.call_args.kwargs)

    def test_source_reference_recursively_exposes_grouped_request_values(self) -> None:
        reference = selected_api_reference(("aurora",), include_policy=False)
        for expected in (
            "Properties: assets: RequestAssets",
            "target: RequestTarget",
            "forecast: ForecastSpecification",
            "policy: DataAccessPolicy",
            "RegionBounds(",
            "RequestProvenance(",
        ):
            self.assertIn(expected, reference)

    def test_source_reference_recursively_exposes_return_value_api(self) -> None:
        reference = selected_api_reference(
            ("sevir", "geolocator", "cascast", "aurora"),
            include_policy=False,
        )
        for expected in (
            "Event.south: float | None",
            "Location.latitude: float",
            "PixelLocation.projected_x: float | None",
            "forecast_member_vil",
            "lead_hour",
        ):
            self.assertIn(expected, reference)


if __name__ == "__main__":
    unittest.main()
