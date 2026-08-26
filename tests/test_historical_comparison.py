import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.tools.sevir import Statistics as statistics
from sevir_world.evaluation.cases import _case_tags


class HistoricalComparisonTests(unittest.TestCase):
    def test_case_tags_are_data_driven(self):
        arrays = {
            "ensemble_members": np.ones((2, 12, 2, 2), dtype=np.float32) * 200,
            "ground_truth": np.ones((12, 2, 2), dtype=np.uint8) * 10,
        }
        tags = _case_tags(arrays)
        self.assertIn("strong_weather", tags)
        self.assertIn("truth_outside_ensemble_range", tags)

    def test_neutral_metrics_exclude_missing_pixels(self):
        frames = np.asarray([[[1, 2], [255, 5]], [[10, 20], [30, 40]]], dtype=np.uint8)
        result = statistics._comparison_metrics(frames, threshold=20)
        self.assertEqual(result["max_vil"]["max"], 40.0)
        self.assertAlmostEqual(result["mean_vil"]["mean"], 13.8333, places=4)
        self.assertEqual(result["per_frame"][0]["valid_pixel_count"], 3)
        self.assertAlmostEqual(result["per_frame"][0]["threshold_coverage"], 0.0, places=5)

    @patch("sevir_world.tools.sevir.Statistics._load_vil_array")
    @patch("sevir_world.tools.sevir.Statistics._catalog_vil_records")
    @patch("sevir_world.tools.sevir.Statistics._catalog_vil_event")
    @patch("sevir_world.tools.sevir.Statistics._load_cascast_ensemble")
    def test_forecast_history_comparison_is_parameterized(
        self, load_ensemble, load_reference, records, load_vil
    ):
        record = {
            "id": "E0", "month": 6, "time_utc": "2020-06-01 00:00:00",
            "event_type": "Unlabeled", "llcrnrlat": 0.0, "llcrnrlon": 0.0,
            "urcrnrlat": 10.0, "urcrnrlon": 10.0,
        }
        candidate = {**record, "id": "E1", "time_utc": "2019-06-01 00:00:00"}
        arrays = {
            "ensemble_mean": np.full((12, 4, 4), 50, dtype=np.float32),
            "ground_truth": np.full((12, 4, 4), 30, dtype=np.uint8),
        }
        load_ensemble.return_value = (arrays, {"event_id": "E0"})
        load_reference.return_value = record
        records.return_value = (record, candidate)
        load_vil.return_value = (np.full((49, 4, 4), 20, dtype=np.uint8), "remote:test")

        result = statistics.compare_cascast_with_historical_region(
            "E0", month_window=0, threshold=40, region=[0, 0, 4, 4]
        )
        self.assertEqual(result["historical"]["sample_count"], 1)
        self.assertEqual(result["forecast"]["metrics"]["max_vil"]["max"], 50.0)
        self.assertEqual(result["historical"]["max_vil"]["median"], 20.0)
        self.assertEqual(result["threshold"], 40.0)


if __name__ == "__main__":
    unittest.main()
