from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.tools.visualization import Visualizer
from sevir_world.tools.uncertainty import UncertaintyAnalyzer


class VisualizationProtocolTests(unittest.TestCase):
    def test_divergence_charts_use_raw_ensemble_and_generic_renderer(self) -> None:
        calls: list[dict] = []

        def recorder(name, arguments, result=None, error=None, cache_hit=False):
            calls.append({"name": name, "arguments": arguments, "result": result})

        records = [
            {
                "seed": seed,
                "lead_minutes": [60],
                "frames": np.full((1, 4, 4), value, dtype=np.float32),
            }
            for seed, value in ((11, 20.0), (22, 30.0), (33, 50.0))
        ]
        raw_payload = {
            "status": "ok",
            "ensemble_members": 3,
            "records": records,
            "selected_lead_minutes": [60],
        }
        with patch(
            "sevir_world.tools.uncertainty.forecast.get_uploaded_ensemble_raw_values",
            return_value=raw_payload,
        ):
            raw = UncertaintyAnalyzer(recorder).uploaded_raw(
                "a" * 32, [60], ensemble_members=3
            )
            cube = np.stack(
                [np.asarray(item["frames"])[0] for item in raw["records"]], axis=0
            )
            member_means = np.mean(cube, axis=(1, 2))
            spatial_std = np.std(cube, axis=0)
            visualizer = Visualizer(recorder, None, None, None)
            bar = visualizer.render_spec({
                "chart_type": "bar",
                "title": "Member mean comparison",
                "labels": ["M1", "M2", "M3"],
                "series": [{"label": "mean", "values": member_means.tolist()}],
            })
            heatmap = visualizer.render_spec({
                "chart_type": "heatmap",
                "title": "Spatial disagreement",
                "panels": [{"label": "standard deviation", "values": spatial_std}],
            })

        self.assertEqual(bar["chart_type"], "agent_chart_spec")
        self.assertEqual(heatmap["chart_type"], "agent_chart_spec")
        self.assertEqual(
            [item["name"] for item in calls],
            [
                "uncertainty.uploaded_raw",
                "visualizer.render_spec",
                "visualizer.render_spec",
            ],
        )
        self.assertEqual(member_means.tolist(), [20.0, 30.0, 50.0])
        self.assertAlmostEqual(float(spatial_std.mean()), 12.472191, places=5)

    def test_render_spec_supports_requested_chart_types(self) -> None:
        calls: list[dict] = []

        def recorder(name, arguments, result=None, error=None, cache_hit=False):
            calls.append({"name": name, "arguments": arguments, "result": result})

        grid = np.arange(16, dtype=np.float32).reshape(4, 4)
        specs = [
            {"chart_type": "raster_map", "panels": [{"label": "观测", "values": grid}]},
            {"chart_type": "bar", "labels": ["A", "B"], "series": [{"label": "值", "values": [1, 2]}]},
            {"chart_type": "grouped_bar", "labels": ["A", "B"], "series": [{"label": "成员1", "values": [1, 2]}, {"label": "成员2", "values": [2, 3]}]},
            {"chart_type": "line", "series": [{"label": "强度", "values": [1, 2]}]},
            {"chart_type": "ensemble_range_line", "series": [{"label": "均值", "values": [1, 2]}], "lower": [0, 1], "upper": [2, 3]},
            {"chart_type": "heatmap", "panels": [{"label": "分歧", "values": grid, "cmap": "viridis"}]},
            {"chart_type": "difference_map", "panels": [{"label": "误差", "values": grid}]},
            {"chart_type": "small_multiples", "panels": [{"label": "成员1", "values": grid}, {"label": "成员2", "values": grid + 1}]},
            {"chart_type": "overlay", "panels": [{"label": "预测", "values": grid}], "overlays": [{"values": grid, "level": 5}]},
            {"chart_type": "trajectory", "tracks": [{"label": "核心", "x": [1, 2], "y": [2, 1]}]},
        ]
        visualizer = Visualizer(recorder, None, None, None)
        results = [visualizer.render_spec(spec) for spec in specs]

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result["verified"] for result in results))
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(call["name"] == "visualizer.render_spec" for call in calls))

    def test_complete_sequence_records_one_tool_call_and_reuses_identical_artifacts(self) -> None:
        calls: list[dict] = []

        def recorder(name, arguments, result=None, error=None, cache_hit=False):
            calls.append({"name": name, "arguments": arguments, "result": result})

        frames = [
            [[frame_index + row + column for column in range(4)] for row in range(4)]
            for frame_index in range(7)
        ]
        labels = [f"+{5 * (index + 1)}min" for index in range(7)]
        visualizer = Visualizer(recorder, None, None, None)
        first = visualizer.forecast_montage_sequence(frames, labels, "Complete sequence")
        second = visualizer.forecast_montage_sequence(frames, labels, "Complete sequence")

        self.assertEqual(first["artifact_count"], 2)
        self.assertEqual(first["frame_count"], 7)
        self.assertEqual(
            [item["artifact_id"] for item in first["artifacts"]],
            [item["artifact_id"] for item in second["artifacts"]],
        )
        self.assertEqual(
            [item["name"] for item in calls],
            [
                "visualizer.forecast_montage_sequence",
                "visualizer.forecast_montage_sequence",
            ],
        )


if __name__ == "__main__":
    unittest.main()
