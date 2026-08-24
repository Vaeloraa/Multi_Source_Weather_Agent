from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.backends.statistics import (
    aggregate_sevir_catalog,
    compare_sevir_catalog_groups,
    compare_event_with_history,
    group_sevir_catalog,
    query_sevir_records,
    rank_sevir_catalog,
    find_similar_event_candidates,
    load_event_history_frame_primitives,
)
from sevir_world.data_access.catalog import load_sevir_vil_frame_histograms


class StatisticsPrimitiveTests(unittest.TestCase):
    def test_local_hdf5_histograms_leave_duration_rule_to_agent(self) -> None:
        record = {
            "id": "R-test",
            "data_max": 180.0,
            "event_type": "Hail",
            "time_utc": "2019-01-01 00:00:00",
            "file_name": "vil/test.h5",
            "file_index": 7,
            "minute_offsets": (-5, 0, 5),
        }
        frames = np.zeros((3, 2, 2), dtype=np.uint8)
        frames[1, 0, 0] = 133
        with (
            patch("sevir_world.data_access.catalog._catalog_vil_event", return_value=record),
            patch(
                "sevir_world.data_access.catalog._load_vil_array",
                return_value=(frames, Path("D:/Profile/专业实习/sevir-agent-demo/data/sevir_complete/data/vil/test.h5")),
            ),
            patch(
                "sevir_world.data_access.catalog.LOCAL_SEVIR_DATA_ROOT",
                Path("D:/Profile/专业实习/sevir-agent-demo/data/sevir_complete/data"),
            ),
        ):
            result = load_sevir_vil_frame_histograms(["R-test"], 0, 3)
        histograms = result["records"][0]["frame_histograms"]
        self.assertEqual(len(histograms), 3)
        self.assertEqual(len(histograms[0]), 256)
        self.assertEqual(sum(histograms[0]), 4)
        self.assertEqual(histograms[1][133], 1)
        self.assertNotIn("duration", result["records"][0])
        self.assertEqual(result["loaded_event_count"], 1)

    def test_event_duration_uses_catalog_hdf5_and_any_pixel_exceedance(self) -> None:
        record = {
            "id": "R-test",
            "data_max": 180.0,
            "event_type": "Unlabeled",
            "time_utc": "2019-01-01 00:00:00",
            "file_name": "vil/test.h5",
            "file_index": 7,
            "minute_offsets": (-5, 0, 5),
        }
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        frames[:, 0, 0] = 133
        with (
            patch(
                "sevir_world.backends.statistics._catalog_vil_records",
                return_value=[record],
            ),
            patch(
                "sevir_world.backends.statistics._load_vil_array",
                return_value=(frames, "vil/test.h5"),
            ),
            patch("sevir_world.backends.statistics._load_cascast_ensemble") as ensemble,
        ):
            result = compare_event_with_history("R-test", threshold=133.0)

        duration = result["observed_high_vil_duration"]
        self.assertTrue(duration["available"])
        self.assertEqual(duration["source"], "official_sevir_hdf5")
        self.assertEqual(duration["longest_consecutive_minutes"], 10)
        self.assertEqual(
            duration["threshold_exceedance_rule"],
            "any_valid_pixel_at_or_above_threshold",
        )
        self.assertNotIn("minimum_region_pixels", duration)
        ensemble.assert_not_called()

    def test_history_primitives_do_not_compute_duration(self) -> None:
        record = {
            "id": "R-test",
            "data_max": 180.0,
            "event_type": "Hail",
            "file_name": "vil/test.h5",
            "file_index": 7,
            "minute_offsets": (-5, 0, 5),
        }
        frames = np.zeros((3, 2, 2), dtype=np.uint8)
        frames[1, 0, 0] = 133
        with (
            patch("sevir_world.backends.statistics._catalog_vil_event", return_value=record),
            patch("sevir_world.backends.statistics._load_vil_array", return_value=(frames, "vil/test.h5")),
        ):
            result = load_event_history_frame_primitives(["R-test"], 0, 3)
        self.assertEqual(result["statistics_computed"], [])
        self.assertEqual([item["max_vil"] for item in result["records"][0]["frames"]], [0, 133, 0])
        self.assertNotIn("duration_minutes", result["records"][0])

    def test_similarity_candidates_expose_rule_without_duration(self) -> None:
        rows = [
            {"id": "R0", "data_max": 150.0, "event_type": "Hail"},
            {"id": "R1", "data_max": 140.0, "event_type": "Hail"},
            {"id": "R2", "data_max": 190.0, "event_type": "Tornado"},
        ]
        reference = {**rows[0], "file_name": "x", "file_index": 0, "minute_offsets": tuple(range(49))}
        with (
            patch("sevir_world.backends.statistics._catalog_vil_event", return_value=reference),
            patch("sevir_world.backends.statistics._catalog_vil_records", return_value=rows),
            patch("sevir_world.backends.statistics.catalog_schema", return_value={"fields": {"data_max": {"numeric": True}}}),
        ):
            result = find_similar_event_candidates(
                "R0", field="data_max", candidate_limit=2,
                same_event_type=True, local_hdf5_only=False,
            )
        self.assertEqual([item["event_id"] for item in result["records"]], ["R1"])
        self.assertTrue(result["similarity_definition"]["duration_not_computed"])
        self.assertFalse(
            result["similarity_definition"]["meteorological_similarity_established"]
        )
        self.assertTrue(
            result["selection_diagnostics"]["selected_field_is_nondiscriminating"]
        )
        self.assertIn("event_histories_raw", result["recommended_next_step"])

    def test_aggregate_returns_only_requested_metric(self) -> None:
        result = aggregate_sevir_catalog(
            year=2019,
            field="data_max",
            operations=["median"],
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("median", result)
        self.assertNotIn("mean", result)
        self.assertNotIn("std", result)
        self.assertIn("valid_value_count", result)

    def test_query_fields_and_limit_are_agent_selected(self) -> None:
        result = query_sevir_records(
            year=2019,
            fields=["id", "data_max"],
            limit=7,
        )
        self.assertEqual(result["returned_count"], 7)
        self.assertTrue(all(set(row) == {"id", "data_max"} for row in result["records"]))

    def test_ranking_limit_is_not_fixed_to_ten(self) -> None:
        result = rank_sevir_catalog(field="data_max", limit=5)
        self.assertEqual(result["limit"], 5)
        self.assertEqual(len(result["records"]), 5)

    def test_dynamic_grouping_and_explicit_group_comparison(self) -> None:
        grouped = group_sevir_catalog(
            group_by="month",
            field="data_max",
            operations=["median"],
            group_limit=3,
        )
        self.assertEqual(grouped["returned_group_count"], 3)
        self.assertTrue(all("median" in item["statistics"] for item in grouped["groups"]))
        compared = compare_sevir_catalog_groups(
            group_by="event_type",
            group_values=["Hail", "Tornado"],
            operations=["median", "std"],
        )
        self.assertEqual([item["group"] for item in compared["groups"]], ["Hail", "Tornado"])
        self.assertTrue(all("std" in item["statistics"] for item in compared["groups"]))


if __name__ == "__main__":
    unittest.main()
