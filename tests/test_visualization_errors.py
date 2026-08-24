from __future__ import annotations

import unittest
from unittest.mock import patch

from sevir_world.tools.visualization import Visualizer


class VisualizationErrorTests(unittest.TestCase):
    def test_grid_panels_records_artifact_write_failure_before_reraising(self) -> None:
        calls: list[dict[str, object]] = []

        def recorder(name, arguments, result=None, error=None, cache_hit=False):
            calls.append({
                "name": name,
                "arguments": arguments,
                "result": result,
                "error": error,
            })

        visualizer = Visualizer(recorder, None, None, None)
        with patch.object(
            visualizer,
            "_save",
            side_effect=PermissionError("artifact directory is not writable"),
        ):
            with self.assertRaisesRegex(PermissionError, "not writable"):
                visualizer.grid_panels([{
                    "label": "平均离散度",
                    "values": [[0.0, 1.0], [2.0, 3.0]],
                    "colorbar_label": "标准差",
                }])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "visualizer.grid_panels")
        self.assertIsInstance(calls[0]["error"], PermissionError)
        self.assertEqual(calls[0]["arguments"]["labels"], ["平均离散度"])


if __name__ == "__main__":
    unittest.main()
