from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.backends.statistics import load_event_history_raw
from sevir_world.tools.metrics import VerificationMetrics
from sevir_world.tools.uncertainty import UncertaintyAnalyzer, track_fields


def _recorder(*_args, **_kwargs) -> None:
    return None


class HistoricalRawPrimitiveTests(unittest.TestCase):
    def test_raw_history_exposes_pixels_timeline_missingness_and_no_statistic(self) -> None:
        record = {
            "id": "R-test",
            "event_type": "Hail",
            "file_name": "vil/test.h5",
            "file_index": 3,
            "minute_offsets": (-5, 0, 5),
        }
        frames = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        frames[1, 0, 0] = 255
        with (
            patch("sevir_world.backends.statistics._catalog_vil_event", return_value=record),
            patch(
                "sevir_world.backends.statistics._load_vil_array",
                return_value=(frames, "vil/test.h5"),
            ),
        ):
            result = load_event_history_raw(["R-test"], 0, 3)

        event = result["records"][0]
        self.assertEqual(event["relative_minutes"], [-5, 0, 5])
        self.assertEqual(event["frame_shape"], [2, 2])
        self.assertEqual(event["missing_pixel_count"], [0, 1, 0])
        self.assertEqual(event["frame_values"][1][0][0], 255)
        self.assertTrue(result["frame_cube_available"])
        self.assertEqual(result["frame_cube"].shape, (1, 3, 2, 2))
        self.assertFalse(result["frame_cube"].flags.writeable)
        self.assertEqual(result["loaded_event_ids"], ["R-test"])
        self.assertEqual(result["relative_minutes"], [-5, 0, 5])
        self.assertEqual(result["statistics_computed"], [])
        self.assertIsNone(result["threshold_applied"])
        self.assertFalse(event["observation_window"]["complete_storm_lifecycle_available"])
        self.assertIn("raw_observed_frames", result["capabilities"])
        self.assertEqual(result["provenance"]["source_kind"], "catalog_indexed_local_sevir_hdf5")

    def test_raw_history_value_budget_fails_per_event_without_truncating_pixels(self) -> None:
        record = {
            "id": "R-test",
            "event_type": "Hail",
            "file_name": "vil/test.h5",
            "file_index": 3,
            "minute_offsets": (-5, 0, 5),
        }
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        with (
            patch("sevir_world.backends.statistics._catalog_vil_event", return_value=record),
            patch(
                "sevir_world.backends.statistics._load_vil_array",
                return_value=(frames, "vil/test.h5"),
            ),
        ):
            result = load_event_history_raw(
                ["R-test"], 0, 3, max_total_values=10
            )
        self.assertFalse(result["records"][0]["available"])
        self.assertIn("value budget", result["records"][0]["error"])
        self.assertEqual(result["loaded_value_count"], 0)

class VerificationRawPrimitiveTests(unittest.TestCase):
    def test_cached_raw_pair_is_aligned_and_exposes_valid_mask(self) -> None:
        prediction = np.asarray(
            [np.full((2, 2), 10.5), np.full((2, 2), 20.5)], dtype=np.float32
        )
        truth = np.asarray(
            [[[10, 255], [12, 13]], [[20, 21], [22, 23]]], dtype=np.uint8
        )
        arrays = {
            "history": np.zeros((13, 2, 2), dtype=np.uint8),
            "ensemble_mean": prediction,
            "ensemble_members": prediction[None],
            "ground_truth": truth,
        }
        metrics = VerificationMetrics(_recorder)
        with patch(
            "sevir_world.tools.metrics.ensemble._load_cascast_ensemble",
            return_value=(arrays, {"cache_file": "event.npz"}),
        ):
            result = metrics.raw_pair("R-test", [5, 10])
        self.assertEqual(result["lead_minutes"], [5, 10])
        self.assertEqual(result["prediction_frames"][0][0][0], 10.5)
        self.assertFalse(result["valid_mask"][0][0][1])
        self.assertEqual(result["valid_pixel_count_by_lead"], [3, 4])
        self.assertEqual(result["statistics_computed"], [])
        self.assertTrue(
            result["provenance"]["truth_was_loaded_explicitly_for_verification"]
        )

    def test_uploaded_raw_pair_loads_truth_only_in_explicit_metrics_call(self) -> None:
        prediction = np.asarray(
            [np.full((2, 2), 10.25), np.full((2, 2), 20.25)], dtype=np.float32
        )
        arrays = {
            "history": np.zeros((13, 2, 2), dtype=np.uint8),
            "ensemble_mean": prediction,
            "ensemble_members": prediction[None],
        }
        truth = {
            "available": True,
            "event_id": "R-match",
            "frames": [np.full((2, 2), 9, dtype=np.uint8).tolist()],
            "source": "held-out.h5",
        }
        metrics = VerificationMetrics(_recorder)
        with (
            patch(
                "sevir_world.tools.metrics.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ),
            patch(
                "sevir_world.tools.metrics.load_uploaded_cascast_ensemble",
                return_value=(arrays, {"cache_file": "upload.npz"}),
            ),
            patch(
                "sevir_world.tools.metrics.load_uploaded_sequence_hdf5_ground_truth",
                return_value=truth,
            ) as load_truth,
        ):
            result = metrics.uploaded_raw_pair(
                "a" * 32, [10], ensemble_members=1
            )
        load_truth.assert_called_once_with("a" * 32, [10])
        self.assertEqual(result["matched_event_id"], "R-match")
        self.assertEqual(result["prediction_frames"][0][0][0], 20.25)
        self.assertEqual(result["lead_minutes"], [10])
        self.assertTrue(result["forecast_evidence_id"].startswith("forecast-"))
        self.assertEqual(result["forecast_evidence"]["ensemble_count"], 1)
        self.assertEqual(result["forecast_evidence"]["horizon_hours"], 1.0)
        self.assertTrue(result["forecast_evidence"]["ground_truth_attached"])


class SpatialCalibrationTests(unittest.TestCase):
    def test_object_tracking_defaults_to_pixels_only(self) -> None:
        fields = np.zeros((2, 8, 8), dtype=np.float32)
        fields[0, 2:4, 2:4] = 200
        fields[1, 2:4, 3:5] = 200
        result = track_fields(fields, threshold=133)
        track = result["tracks"][0]
        self.assertIn("path_length_pixels", track)
        self.assertNotIn("path_length_km", track)
        self.assertNotIn("area_km2", track["observations"][0])
        self.assertFalse(result["provenance"]["physical_calibration_available"])
        self.assertFalse(result["provenance"]["compass_direction_available"])

    def test_physical_units_require_named_calibration_source(self) -> None:
        fields = np.zeros((2, 4, 4), dtype=np.float32)
        fields[:, 1:3, 1:3] = 200
        with self.assertRaisesRegex(ValueError, "calibration_source"):
            track_fields(fields, threshold=133, pixel_size_km=2.0)
        result = track_fields(
            fields,
            threshold=133,
            pixel_size_km=2.0,
            calibration_source="event projection metadata",
        )
        self.assertIn("path_length_km", result["tracks"][0])
        self.assertIn("area_km2", result["tracks"][0]["observations"][0])

    def test_uncertainty_upload_does_not_request_truth_by_default(self) -> None:
        analyzer = UncertaintyAnalyzer(_recorder)
        expected = {"status": "ok", "ground_truth_available": False}
        with patch(
            "sevir_world.tools.uncertainty.forecast.get_uploaded_ensemble_raw_values",
            return_value=expected,
        ) as backend:
            result = analyzer.uploaded_raw("a" * 32, [60], ensemble_members=2)
        backend.assert_called_once_with(
            upload_id="a" * 32,
            lead_minutes=[60],
            ensemble_members=2,
            horizon_hours=1.0,
            include_ground_truth=False,
            array_transport=True,
            force_live=False,
        )
        self.assertFalse(result["ground_truth_available"])
        self.assertFalse(
            result["ensemble_semantics"]["calibrated_probability_available"]
        )
        self.assertFalse(result["decision_capabilities"]["route_safety"])
        self.assertIn("not a calibrated chance", result["interpretation_limit"])

    def test_uncertainty_accepts_all_twelve_one_hour_leads_in_one_agent_call(self) -> None:
        analyzer = UncertaintyAnalyzer(_recorder)

        def backend(**arguments):
            leads = arguments["lead_minutes"]
            records = []
            for member in range(2):
                frames = np.full(
                    (len(leads), 2, 2), member + leads[0], dtype=np.float32
                )
                frames.setflags(write=False)
                records.append({
                    "member": member + 1,
                    "seed": member,
                    "lead_minutes": leads,
                    "frames": frames,
                })
            return {
                "status": "ok",
                "capabilities": ["ensemble_members"],
                "forecast_evidence_id": "forecast-test",
                "forecast_evidence": {"id": "forecast-test"},
                "ensemble_members": 2,
                "records": records,
                "cache_hit": True,
                "ground_truth_available": False,
                "ground_truth": {"available": False, "frames": []},
                "provenance": {},
            }

        leads = list(range(5, 65, 5))
        with patch(
            "sevir_world.tools.uncertainty.forecast.get_uploaded_ensemble_raw_values",
            side_effect=backend,
        ) as raw_backend:
            result = analyzer.uploaded_raw("a" * 32, leads, ensemble_members=2)
        self.assertEqual(raw_backend.call_count, 3)
        self.assertEqual(result["backend_chunk_count"], 3)
        self.assertEqual(result["selected_lead_minutes"], leads)
        self.assertEqual(result["records"][0]["frames"].shape, (12, 2, 2))
        self.assertFalse(result["records"][0]["frames"].flags.writeable)
        self.assertEqual(
            result["forecast_evidence"]["selected_lead_minutes"], leads
        )


if __name__ == "__main__":
    unittest.main()
