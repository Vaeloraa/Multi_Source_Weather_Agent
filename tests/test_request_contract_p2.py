from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.capabilities.base import RequestBinding
from sevir_world.api.unified_service import bind_request_assets
from sevir_world.agent.tool_docs import selected_api_reference
from sevir_world.tools._shared import event_value, observation_value
from sevir_world.tools.cascast.CasCast import CasCast
from sevir_world.tools.sevir.Sevir import Sevir


def _recorder(*_args, **_kwargs) -> None:
    return None


def _raw_forecast(initialization_time_utc: str | None = None) -> dict:
    return {
        "status": "ok",
        "upload_id": "upload-a",
        "model_name": "cascast_live",
        "initialization_time_utc": initialization_time_utc,
        "history_frames": np.zeros((13, 2, 2), dtype=np.float32),
        "history_relative_minutes": list(range(-60, 5, 5)),
        "forecast_frames": np.zeros((1, 2, 2), dtype=np.float32),
        "forecast_relative_minutes": [5],
        "ensemble": {},
        "cache_hit": True,
    }


class RequestContractP2Tests(unittest.TestCase):
    def test_public_time_adapters_always_emit_iso_utc(self) -> None:
        event = event_value({"record": {
            "id": "r1",
            "time_utc": "2019-06-27 08:58:20",
            "event_type": "test",
            "frame_times_utc": ["2019-06-27 08:58:20"],
            "minute_offsets": [0],
        }})
        observation = observation_value({
            "event_id": "r1",
            "event_anchor_time_utc": "2019-06-27 08:58:20",
            "frame_times_utc": ["2019-06-27 08:58:20"],
            "values": np.zeros((1, 2, 2)),
            "relative_minutes": [0],
        })
        for value in (
            event.event_anchor_time_utc,
            event.frame_times_utc[0],
            observation.attrs["event_anchor_time_utc"],
            observation.coords["time_utc"].values[0],
        ):
            self.assertRegex(str(value), r"T\d{2}:\d{2}:\d{2}Z$")

    def test_search_result_preserves_every_applied_public_filter(self) -> None:
        with patch(
            "sevir_world.tools.sevir.Catalog.search_sevir_catalog",
            return_value={
                "status": "ok",
                "records": [],
                "total_matches": 0,
                "returned_count": 0,
            },
        ) as backend:
            result = Sevir(_recorder, RequestBinding()).search(
                require_prior_minutes=60,
            )
        self.assertEqual(result.attrs["criteria"]["require_prior_minutes"], 60)
        self.assertNotIn("require_observation_cache", backend.call_args.kwargs)
        self.assertNotIn("require_observation_cache", inspect.getdoc(Sevir.search) or "")

    def test_unmatched_upload_does_not_use_question_time_as_model_time(self) -> None:
        binding = RequestBinding.from_mapping({
            "upload_id": "upload-a",
            "requested_time_utc": "2024-01-01T00:00:00Z",
        })
        with patch(
            "sevir_world.tools.cascast.Forecast.get_uploaded_vil_sequence_forecast",
            return_value=_raw_forecast(),
        ):
            forecast = CasCast(_recorder, binding).forecast()
        self.assertIsNone(forecast.attrs["initialization_time_utc"])
        self.assertTrue(all(not value for value in forecast.coords["valid_time_utc"].values))

    def test_upload_metadata_time_is_normalized_and_drives_valid_times(self) -> None:
        with patch(
            "sevir_world.tools.cascast.Forecast.get_uploaded_vil_sequence_forecast",
            return_value=_raw_forecast("2019-06-27 08:58:20"),
        ):
            forecast = CasCast(
                _recorder,
                RequestBinding(upload_id="upload-a"),
            ).forecast()
        self.assertEqual(forecast.attrs["initialization_time_utc"], "2019-06-27T08:58:20Z")
        self.assertEqual(forecast.coords["valid_time_utc"].values[0], "2019-06-27T09:03:20Z")

    def test_web_request_uses_only_canonical_request_field_names(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "web" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("requested_time_utc: normalizedAnalysisTimeUtc()", source)
        self.assertIn("forecast_leads_hours: [6, 12, 18, 24]", source)
        self.assertIn("upload_id: uploadedSequenceId || uploadedSampleId", source)
        self.assertNotIn("analysis_time_utc: normalizedAnalysisTimeUtc()", source)
        self.assertNotIn("forecast_leads: [6, 12, 18, 24]", source)
        self.assertNotIn("sequence_upload_id: uploadedSequenceId", source)

    def test_explicit_event_cannot_conflict_with_upload_metadata(self) -> None:
        payload = {"event_id": "R1", "upload_id": "upload-a"}
        with patch(
            "sevir_world.api.unified_service.uploaded_event_match",
            return_value={"event_match": {"status": "matched", "event_id": "R2"}},
        ), self.assertRaisesRegex(ValueError, "conflicts with upload event_id"):
            bind_request_assets("test", payload)

    def test_stale_study_event_is_overridden_by_current_upload(self) -> None:
        payload = {
            "event_id": "R1",
            "upload_id": "upload-a",
            "event_resolution": {
                "source": "study_case_decision",
                "event_id": "R1",
            },
        }
        with patch(
            "sevir_world.api.unified_service.uploaded_event_match",
            return_value={"event_match": {"status": "matched", "event_id": "R2"}},
        ), patch(
            "sevir_world.tools.sevir.Catalog.resolve_sevir_subject_key",
            return_value="sevir:r2",
        ):
            bind_request_assets("test", payload)
        assert payload["event_id"] == "R2"
        assert payload["event_resolution"]["source"] == "bound_upload_overrode_stale_study_case"


if __name__ == "__main__":
    unittest.main()
