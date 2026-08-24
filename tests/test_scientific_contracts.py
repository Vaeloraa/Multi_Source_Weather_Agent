from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.api.tool_gateway import TOOL_FUNCTIONS
from sevir_world.data_access.uploads import event_match_from_metadata
from sevir_world.data_access.uploads import (
    load_uploaded_sequence_hdf5_ground_truth,
    match_uploaded_sequence_hdf5,
)
from sevir_world.backends.forecast import (
    _scientific_contract,
    analyze_uploaded_vil,
    get_uploaded_vil_sequence_forecast,
    query_uploaded_history_values,
    query_uploaded_ensemble_member_values,
    get_uploaded_ensemble_raw_values,
)
from sevir_world.tools.uncertainty import UncertaintyAnalyzer


class ScientificContractTests(unittest.TestCase):
    def test_upload_event_match_uses_explicit_metadata_only(self) -> None:
        unmatched = event_match_from_metadata("weather.png")
        self.assertEqual(unmatched["status"], "not_provided")
        matched = event_match_from_metadata("R19082500047459_frame_13.png")
        self.assertEqual(matched["status"], "matched")
        self.assertEqual(matched["event_id"], "R19082500047459")
        self.assertEqual(matched["method"], "filename")

    def test_event_match_tool_is_registered(self) -> None:
        self.assertIn("match_uploaded_event", TOOL_FUNCTIONS)
    def test_contract_disables_metrics_without_ground_truth(self) -> None:
        contract = _scientific_contract(
            inference_type="observation_only",
            ground_truth_available=False,
            input_frames=1,
            forecast_frames=0,
            lead_minutes=0,
        )
        self.assertFalse(contract["metrics_available"])
        self.assertEqual(contract["forecast_frame_count"], 0)

    def test_generic_statistics_are_registered_for_api_agent(self) -> None:
        for name in (
            "query_sevir_records",
            "aggregate_sevir_catalog",
            "group_sevir_catalog",
            "compare_sevir_catalog_groups",
            "rank_sevir_catalog",
        ):
            self.assertIn(name, TOOL_FUNCTIONS)

    def test_ensemble_support_uses_runtime_member_count(self) -> None:
        members = np.zeros((3, 2, 8, 8), dtype=np.float32)
        members[:, :, 2:5, 2:5] = 200.0
        arrays = {"ensemble_members": members}
        analyzer = UncertaintyAnalyzer(lambda *_args, **_kwargs: None)
        with patch(
            "sevir_world.tools.uncertainty.ensemble._load_cascast_ensemble",
            return_value=(arrays, {}),
        ):
            result = analyzer.match_ensemble_objects(
                "test-event", threshold=133.0, minimum_region_pixels=1
            )
        self.assertEqual(result["ensemble_count"], 3)
        self.assertTrue(
            all(item["member_support_fraction"] <= 1 for item in result["consensus_objects"])
        )

    def test_uploaded_ensemble_query_returns_only_selected_primitives(self) -> None:
        members = np.zeros((3, 2, 8, 8), dtype=np.float32)
        members[0, :, 1:4, 1:4] = 180.0
        members[1, :, 2:5, 2:5] = 200.0
        members[2, :, 3:6, 3:6] = 220.0
        arrays = {
            "history": np.zeros((13, 8, 8), dtype=np.float32),
            "ensemble_members": members,
            "member_seed": np.asarray([11, 22, 33]),
        }
        metadata = {
            "upload_id": "test-upload",
            "cache_file": "test-upload.npz",
        }
        with (
            patch(
                "sevir_world.backends.forecast.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ),
            patch(
                "sevir_world.backends.forecast.load_uploaded_cascast_ensemble",
                return_value=(arrays, metadata),
            ),
        ):
            result = query_uploaded_ensemble_member_values(
                "test-upload",
                fields=["max_vil", "exceeding_area_ratio"],
                lead_minutes=[5, 10],
                threshold=160.0,
                minimum_region_pixels=1,
                ensemble_members=3,
            )
        self.assertEqual(result["ensemble_count"], 3)
        self.assertEqual(result["record_count"], 6)
        self.assertFalse(result["ground_truth_available"])
        self.assertEqual(result["selected_fields"], ["max_vil", "exceeding_area_ratio"])
        self.assertEqual(result["selected_lead_minutes"], [5, 10])
        self.assertEqual(result["calculated_statistics"], [])
        self.assertTrue(all("max_vil" in item for item in result["records"]))
        self.assertTrue(all("mean_vil" not in item for item in result["records"]))
        self.assertNotIn("spatial_frame_summaries", result)

    def test_uploaded_raw_has_documented_member_shape_and_matched_truth(self) -> None:
        members = np.zeros((2, 12, 8, 8), dtype=np.float32)
        members[0, 11, 0, 0] = 1.75
        members[0, 11, 1:3, 1:3] = 180
        members[1, 11, 2:4, 2:4] = 220
        arrays = {
            "ensemble_members": members,
            "member_seed": np.asarray([11, 22]),
        }
        truth = {
            "available": True,
            "event_id": "R-test",
            "lead_minutes": [60],
            "frames": [np.zeros((8, 8), dtype=np.uint8).tolist()],
        }
        with (
            patch(
                "sevir_world.backends.forecast.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ),
            patch(
                "sevir_world.backends.forecast.load_uploaded_cascast_ensemble",
                return_value=(arrays, {}),
            ),
            patch(
                "sevir_world.backends.forecast.load_uploaded_sequence_hdf5_ground_truth",
                return_value=truth,
            ),
        ):
            result = get_uploaded_ensemble_raw_values(
                "test-upload", lead_minutes=[60], ensemble_members=2,
                include_ground_truth=True,
            )
        self.assertNotIn("lead_60", result)
        self.assertEqual(result["selected_lead_minutes"], [60])
        self.assertEqual(result["records"][0]["lead_minutes"], [60])
        self.assertEqual(len(result["records"][0]["frames"][0]), 8)
        self.assertEqual(len(result["records"][0]["frames"][0][0]), 8)
        self.assertEqual(result["dtype"], "float32")
        self.assertAlmostEqual(result["records"][0]["frames"][0][0][0], 1.75)
        self.assertFalse(result["records"][0]["is_deterministic"])
        self.assertEqual(result["output_type"], "stochastic_ensemble")
        self.assertTrue(result["ground_truth_available"])
        self.assertEqual(result["ground_truth"]["event_id"], "R-test")

    def test_uploaded_raw_accepts_two_hour_leads_and_passes_rolling_segment(self) -> None:
        members = np.zeros((2, 24, 8, 8), dtype=np.float32)
        arrays = {"ensemble_members": members, "member_seed": np.asarray([11, 22])}
        truth = {"available": False, "lead_minutes": [65, 120], "frames": []}
        with (
            patch(
                "sevir_world.backends.forecast.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ) as run_inference,
            patch(
                "sevir_world.backends.forecast.load_uploaded_cascast_ensemble",
                return_value=(arrays, {"cache_file": "roll.npz"}),
            ) as load_ensemble,
            patch(
                "sevir_world.backends.forecast.load_uploaded_sequence_hdf5_ground_truth",
                return_value=truth,
            ),
        ):
            result = get_uploaded_ensemble_raw_values(
                "test-upload", lead_minutes=[65, 120], ensemble_members=2, horizon_hours=2.0
            )
        run_inference.assert_called_once_with(
            "test-upload", ensemble_members=2, rolling_segments=2, force=False
        )
        load_ensemble.assert_called_once_with(
            "test-upload", ensemble_members=2, rolling_segments=2
        )
        self.assertEqual(result["horizon_hours"], 2.0)
        self.assertEqual(result["rolling_segments"], 2)
        self.assertEqual(result["selected_lead_minutes"], [65, 120])

    def test_current_example_upload_matches_hdf5_and_loads_plus_60_truth(self) -> None:
        upload_id = "2ddc78931bda4e169038369624d6f641"
        try:
            match = match_uploaded_sequence_hdf5(upload_id)
        except ValueError:
            self.skipTest("persisted UI upload is not present in this environment")
        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["event_id"], "R19062708588472")
        self.assertGreaterEqual(match["indexed_rows"], 2324)
        truth = load_uploaded_sequence_hdf5_ground_truth(upload_id, [60])
        self.assertTrue(truth["available"])
        self.assertEqual(truth["lead_minutes"], [60])
        self.assertEqual(truth["hdf5_frame_indices"], [36])
        self.assertEqual(len(truth["frames"][0]), 384)

    def test_uploaded_cascast_supports_recursive_two_hour_horizon(self) -> None:
        sequence = np.zeros((13, 8, 8), dtype=np.uint8)
        members = np.zeros((2, 24, 8, 8), dtype=np.float32)
        arrays = {
            "history": sequence,
            "ensemble_members": members,
            "ensemble_mean": members.mean(axis=0),
            "member_seed": np.asarray([1, 2]),
        }
        metadata = {"upload_id": "test-upload", "cache_file": "test-roll.npz"}
        with (
            patch(
                "sevir_world.backends.forecast._load_uploaded_vil_sequence",
                return_value=(sequence, {"history_minutes": list(range(-60, 5, 5)), "source": "test"}),
            ),
            patch(
                "sevir_world.backends.forecast.run_uploaded_cascast_inference",
                return_value={"cache_hit": True},
            ),
            patch(
                "sevir_world.backends.forecast.load_uploaded_cascast_ensemble",
                return_value=(arrays, metadata),
            ),
        ):
            result = get_uploaded_vil_sequence_forecast(
                "test-upload",
                model_name="cascast_live",
                ensemble_members=2,
                horizon_hours=2.0,
            )
        self.assertEqual(result["forecast_frame_count"], 24)
        self.assertEqual(result["lead_minutes"], 120)
        self.assertEqual(result["rolling_segments"], 2)
        self.assertEqual(result["forecast_relative_minutes"][-1], 120)

    def test_uploaded_history_query_is_observation_only_and_field_selected(self) -> None:
        sequence = np.zeros((13, 8, 8), dtype=np.uint8)
        sequence[0, 1:4, 1:4] = 180
        sequence[-1, 2:6, 2:6] = 200
        metadata = {
            "source": "test_uploaded_history",
            "history_minutes": list(range(-60, 5, 5)),
        }
        with patch(
            "sevir_world.backends.forecast._load_uploaded_vil_sequence",
            return_value=(sequence, metadata),
        ):
            result = query_uploaded_history_values(
                "test-upload",
                fields=["max_vil", "exceeding_area_ratio"],
                relative_minutes=[-60, 0],
                threshold=160.0,
                minimum_region_pixels=1,
            )
        self.assertEqual(result["forecast_frame_count"], 0)
        self.assertEqual(result["history_elapsed_minutes"], 60)
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["calculated_change_statistics"], [])
        self.assertTrue(all("mean_vil" not in item for item in result["records"]))
        self.assertFalse(result["tracks_included"])

    def test_single_upload_does_not_claim_physical_area_calibration(self) -> None:
        frame = np.zeros((8, 8), dtype=np.uint8)
        frame[1:3, 1:3] = 200
        metadata = {
            "filename": "frame.png",
            "source": "test_upload",
            "grayscale_source": True,
        }
        with patch(
            "sevir_world.backends.forecast._load_uploaded_vil",
            return_value=(frame, metadata),
        ):
            result = analyze_uploaded_vil(
                "test-upload", threshold=160.0, minimum_region_pixels=1
            )
        self.assertFalse(result["physical_area_available"])
        self.assertIsNone(result["pixel_area_km2"])
        self.assertEqual(result["area_calibration"], "unavailable_for_standalone_upload")
        self.assertNotIn("area_sq_km", result)

    def test_uploaded_history_track_duration_uses_elapsed_minutes(self) -> None:
        sequence = np.zeros((13, 8, 8), dtype=np.uint8)
        sequence[:, 2:5, 2:5] = 200
        metadata = {
            "source": "test_uploaded_history",
            "history_minutes": list(range(-60, 5, 5)),
        }
        with patch(
            "sevir_world.backends.forecast._load_uploaded_vil_sequence",
            return_value=(sequence, metadata),
        ):
            result = query_uploaded_history_values(
                "test-upload",
                fields=["max_vil"],
                relative_minutes=[-60, 0],
                threshold=160.0,
                minimum_region_pixels=1,
                include_tracks=True,
            )
        track = result["tracking"]["tracks"][0]
        self.assertEqual(track["first_observed_relative_minute"], -60)
        self.assertEqual(track["last_observed_relative_minute"], 0)
        self.assertEqual(track["observed_span_minutes"], 60)


if __name__ == "__main__":
    unittest.main()
