from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.tools.geolocator import Geospatial as geospatial


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_FIXTURE = ROOT / "data" / "uploads" / "26b6ce621bc8494681b7dfc83cf48de1.npz"


def _focus_pixels() -> list[list[int]]:
    values = np.load(UPLOAD_FIXTURE, allow_pickle=False)["pixels"]
    mask = (values != 255) & (values >= 133)
    valid = values != 255
    peak = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(np.where(valid, values.astype(np.float32), -1.0))), values.shape
        )
    )
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(mask):
        start = (int(row), int(column))
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        components.append(component)
    focus = next(component for component in components if peak in component)
    return [[row, column] for row, column in focus]


@unittest.skipUnless(UPLOAD_FIXTURE.exists(), "real annotation fixture is unavailable")
class VectorizedRegionTests(unittest.TestCase):
    def test_projection_context_is_reused_for_the_same_event_grid(self) -> None:
        record = geospatial._catalog_vil_event("R19062708588472")
        geospatial._projection_context.cache_clear()
        first = geospatial._pixel_centres_latlon(record, [110, 155], [97, 158])
        second = geospatial._pixel_centres_latlon(record, [180], [237])
        cache = geospatial._projection_context.cache_info()
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(np.asarray(first["latitudes"]).shape, (2,))
        self.assertEqual(np.asarray(second["longitudes"]).shape, (1,))

    def test_real_3153_pixel_region_keeps_exact_counties_under_budget(self) -> None:
        pixels = _focus_pixels()
        self.assertEqual(len(pixels), 3153)
        started = time.perf_counter()
        with patch.object(
            geospatial,
            "reverse_geocode_us",
            side_effect=AssertionError("scalar reverse geocoding must not be used"),
        ):
            result = geospatial.geolocate_pixel_region("R19062708588472", pixels)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0)
        self.assertEqual(result["pixel_bbox"], [110, 97, 180, 237])
        self.assertEqual(
            result["latlon_bbox"],
            {"south": 47.165225, "west": -101.310921, "north": 47.810337, "east": -99.464877},
        )
        self.assertEqual(result["provenance"]["projection_transform_batches"], 1)
        self.assertEqual(result["provenance"]["candidate_county_count"], 8)
        self.assertEqual(result["provenance"]["matched_county_count"], 6)
        self.assertEqual(
            [(item["county"], item["pixel_count"]) for item in result["administrative_coverage"]],
            [
                ("Sheridan", 1361),
                ("McLean", 953),
                ("Wells", 575),
                ("Burleigh", 215),
                ("Oliver", 45),
                ("Mercer", 4),
            ],
        )

    def test_duplicate_pixels_are_counted_once(self) -> None:
        result = geospatial.geolocate_pixel_region(
            "R19062708588472", [[155, 158], [155, 158], [156, 158]]
        )
        self.assertEqual(result["pixel_count"], 2)


if __name__ == "__main__":
    unittest.main()
