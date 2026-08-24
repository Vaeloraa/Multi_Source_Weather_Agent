from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sevir_world.tools import visualization
from sevir_world.tools.visualization import Visualizer


class AnnotationRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        visualization._RENDER_CACHE.clear()

    def test_renderer_draws_agent_supplied_peak_and_box(self) -> None:
        values = np.zeros((384, 384), dtype=np.float32)
        spec = {
            "chart_type": "overlay",
            "title": "annotation contract",
            "panels": [{"label": "VIL", "values": values, "cmap": "cascast_vil"}],
            "overlays": [{"values": np.zeros_like(values), "level": 0.5}],
            "annotations": {
                "points": [{"row": 155, "column": 158, "radius": 6, "width": 3}],
                "boxes": [{"left": 97, "top": 110, "right": 237, "bottom": 180, "width": 3}],
            },
        }
        result = Visualizer(lambda *args, **kwargs: None, None, None, None).render_spec(spec)
        path = visualization.ARTIFACT_ROOT / f"{result['artifact_id']}.png"
        with Image.open(path).convert("RGB") as image:
            rendered = np.asarray(image)
        red = (
            (rendered[:, :, 0] > 180)
            & (rendered[:, :, 1] < 100)
            & (rendered[:, :, 2] < 100)
        )
        self.assertTrue(np.any(red[36 + 148:36 + 163, 151:166]))
        self.assertTrue(np.any(red[36 + 109:36 + 113, 97:238]))
        self.assertTrue(result["verified"])


if __name__ == "__main__":
    unittest.main()
