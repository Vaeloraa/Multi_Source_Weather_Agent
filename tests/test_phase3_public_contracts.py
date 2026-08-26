from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.capabilities.base import RegionBounds, RequestBinding
from sevir_world.code_execution.client import ExecutionClient
from sevir_world.tools.aurora.Aurora import Aurora
from sevir_world.tools.cascast.CasCast import CasCast
from sevir_world.tools.sevir.Catalog import (
    _catalog_subject_key,
    _catalog_vil_event,
    resolve_sevir_subject_key,
)
from sevir_world.tools.sevir.Sevir import Sevir
from sevir_world.agent.study import apply_study_case_context
from sevir_world.agent.runtime import ToolUsageRecorder


EVENT_ID = "R19062708588472"


def _recorder(*_args, **_kwargs) -> None:
    return None


class Phase3PublicContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subject_key = resolve_sevir_subject_key(EVENT_ID)
        self.binding = RequestBinding(
            event_id=EVENT_ID,
            sevir_subject_key=self.subject_key,
        )

    def test_exact_subject_identity_and_continuous_hour_compose(self) -> None:
        sevir = Sevir(_recorder, self.binding)
        found = sevir.search(query=EVENT_ID, require_prior_minutes=60)
        self.assertEqual(found.iloc[0]["subject_key"], self.subject_key)
        self.assertNotEqual(self.subject_key, EVENT_ID.lower())
        self.assertTrue(bool(found.iloc[0]["history_is_continuous"]))
        self.assertEqual(int(found.iloc[0]["history_expected_frames"]), 13)
        observed = sevir.observations(minutes_before=60, minutes_after=0)
        self.assertEqual(observed.sizes["time"], 13)
        self.assertEqual(observed.minute_offset.values.tolist(), list(range(-60, 5, 5)))
        self.assertEqual(observed.attrs["subject_key"], self.subject_key)

    def test_cascast_exposes_identity_missing_values_and_forecast_mode(self) -> None:
        forecast = CasCast(_recorder, self.binding).forecast(
            horizon_hours=1.0,
            ensemble_members=2,
            include_ground_truth=True,
        )
        self.assertEqual(forecast.attrs["subject_key"], self.subject_key)
        self.assertEqual(forecast.attrs["subject_kind"], "sevir_event")
        self.assertIn("forecast_mode", forecast.attrs)
        self.assertIn("uses_future_observations", forecast.attrs)
        self.assertFalse(bool(forecast.attrs["uses_future_observations"]))
        if "ground_truth_vil" in forecast:
            self.assertFalse(bool(np.any(forecast["ground_truth_vil"].values == 255)))

    def test_aurora_returns_every_requested_lead_and_availability(self) -> None:
        binding = RequestBinding(
            event_id=EVENT_ID,
            aurora_initialization_time_utc="2019-06-27T03:00:00Z",
            aurora_region=RegionBounds(41.3, -111.98, 53.05, -88.23),
            forecast_leads_hours=(6, 12, 18, 24),
            history_before_hours=6,
            history_after_hours=24,
        )
        environment = Aurora(_recorder, binding).regional_fields()
        self.assertEqual(environment.lead_hour.values.tolist(), [0, 6, 12, 18, 24])
        self.assertEqual(environment.attrs["requested_leads_hours"], (6, 12, 18, 24))
        self.assertEqual(environment.attrs["missing_leads_hours"], ())
        self.assertEqual(environment.attrs["requested_region"], environment.attrs["effective_region"])
        self.assertIn("verification_temperature_2m_c", environment)

    def test_failed_worker_step_does_not_commit_partial_variables(self) -> None:
        client = ExecutionClient(timeout_seconds=10, live_timeout_seconds=10)
        try:
            first = client.execute(
                "x = 1\nresult = {'x': x}", {}, selected_tools=["ds"]
            )
            failed = client.execute(
                "x = 2\nfield = np.zeros((2, 3))\nraise RuntimeError('stop')\nafter = 3",
                {},
                selected_tools=["ds"],
            )
            final = client.execute(
                "result = {'x': x, 'field_exists': 'field' in globals(), "
                "'after_exists': 'after' in globals()}",
                {},
                selected_tools=["ds"],
            )
        finally:
            client.close()
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["namespace_state"]["transaction"], "committed")
        self.assertEqual(first["namespace_state"]["safe_to_reuse"], ["x"])
        self.assertEqual(first["namespace_state"]["result_slot"], "committed")
        self.assertEqual(failed["status"], "error")
        state = failed["namespace_state"]
        self.assertEqual(state["transaction"], "rolled_back")
        self.assertEqual(state["safe_to_reuse"], ["x"])
        self.assertEqual(state["available_values"]["x"]["type"], "int")
        self.assertIn("x", state["materialized_uncommitted_names"])
        self.assertIn("field", state["materialized_uncommitted_names"])
        self.assertIn("after", state["not_reached_assignment_names"])
        self.assertNotIn("result", failed["available_names"])
        self.assertEqual(final["result"]["x"], 1)
        self.assertFalse(final["result"]["field_exists"])
        self.assertFalse(final["result"]["after_exists"])

    def test_failed_worker_reports_array_shapes_without_committing_values(self) -> None:
        client = ExecutionClient(timeout_seconds=10, live_timeout_seconds=10)
        try:
            failed = client.execute(
                "field = np.zeros((3, 48, 96))\nraise ValueError('plot failed')",
                {},
                selected_tools=["ds"],
            )
            after = client.execute(
                "result = {'field_exists': 'field' in globals()}",
                {},
                selected_tools=["ds"],
            )
        finally:
            client.close()
        self.assertEqual(failed["failed_value_shapes"]["field"]["shape"], [3, 48, 96])
        self.assertFalse(failed["failed_value_shapes"]["field"]["committed"])
        self.assertFalse(after["result"]["field_exists"])

    def test_prior_request_facts_do_not_become_new_request_facts(self) -> None:
        payload: dict = {}
        state = {"study_case": {
            "decisions": {},
            "records": [{"request_facts": {
                "requested_time_utc": "2019-06-27T03:00:00Z",
                "region": {"south": 41, "west": -112, "north": 53, "east": -88},
            }}],
        }}
        apply_study_case_context(payload, state)
        self.assertEqual(payload, {})

    def test_promoted_subject_binds_event_and_exact_key_together(self) -> None:
        payload: dict = {}
        state = {"study_case": {"decisions": {"selected_subject": {
            "subject_key": self.subject_key,
            "event_id": EVENT_ID,
        }}}}
        apply_study_case_context(payload, state)
        self.assertEqual(payload["event_id"], EVENT_ID)
        self.assertEqual(payload["sevir_subject_key"], self.subject_key)

    def test_duplicate_event_ids_require_exact_subject_identity(self) -> None:
        first = {"id": "R-DUP", "file_name": "a.h5", "file_index": 7}
        second = {"id": "R-DUP", "file_name": "b.h5", "file_index": 9}
        first_key = _catalog_subject_key(first)
        second_key = _catalog_subject_key(second)
        self.assertNotEqual(first_key, second_key)
        with (
            patch(
                "sevir_world.tools.sevir.Catalog._catalog_vil_index",
                return_value={"r-dup": (first, second)},
            ),
            patch(
                "sevir_world.tools.sevir.Catalog._catalog_subject_index",
                return_value={first_key: first, second_key: second},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "2 CATALOG records"):
                _catalog_vil_event("R-DUP")
            self.assertIs(_catalog_vil_event(first_key), first)
            self.assertIs(_catalog_vil_event(second_key), second)

    def test_invalid_batch_is_rejected_before_tool_cost_is_reserved(self) -> None:
        recorder = ToolUsageRecorder(cost_budget_units=30)
        sevir = Sevir(recorder, self.binding)
        with self.assertRaisesRegex(ValueError, "limited to 20"):
            sevir.observation_batch([f"sevir:{index:024x}" for index in range(21)])
        self.assertEqual(recorder.cost_used_units, 0.0)
        self.assertEqual(recorder.calls, [])

if __name__ == "__main__":
    unittest.main()
