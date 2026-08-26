from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from sevir_world.tools.geolocator import Geospatial as geospatial


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class GeocoderContractTests(unittest.TestCase):
    def test_explicit_coordinates_are_resolved_without_network(self) -> None:
        with patch.object(geospatial, "reverse_geocode_us", return_value={"status": "ok"}):
            result = geospatial.resolve_place("35.0, -118.0")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provenance"]["source"], "explicit_wgs84_coordinates")

    def test_place_name_preserves_ambiguous_candidates(self) -> None:
        payload = [
            {"lat": "35.0", "lon": "-118.0", "display_name": "A"},
            {"lat": "36.0", "lon": "-119.0", "display_name": "B"},
        ]
        with patch("urllib.request.urlopen", return_value=_Response(payload)):
            result = geospatial.resolve_place("Springfield")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)

    def test_region_coverage_distinguishes_region_from_centroid(self) -> None:
        transformed = {
            "rows": np.asarray([0, 1, 0]),
            "columns": np.asarray([0, 3, 2]),
            "latitudes": np.asarray([30.0, 31.0, 30.0]),
            "longitudes": np.asarray([-100.0, -97.0, -98.0]),
            "projected_x_m": np.asarray([0.0, 3000.0, 2000.0]),
            "projected_y_m": np.asarray([0.0, -1000.0, 0.0]),
            "pixel_width_m": 1000.0,
            "pixel_height_m": 1000.0,
        }
        admins = [
            {"status": "ok", "country": "US", "state": "S", "state_code": "SS", "county": "A", "county_fips": "a"},
            {"status": "ok", "country": "US", "state": "S", "state_code": "SS", "county": "B", "county_fips": "b"},
            {"status": "ok", "country": "US", "state": "S", "state_code": "SS", "county": "B", "county_fips": "b"},
        ]
        with patch.object(geospatial, "_catalog_vil_event", return_value={
            "id": "R1", "size_x": 4, "size_y": 4, "width_m": 4000,
            "height_m": 4000, "proj": "+proj=longlat +datum=WGS84 +no_defs",
            "llcrnrlon": -100, "llcrnrlat": 30, "urcrnrlon": -96, "urcrnrlat": 34,
        }), patch.object(geospatial, "_pixel_centres_latlon", return_value=transformed), patch.object(
            geospatial, "_batch_reverse_geocode_us", return_value=(admins, {
                "candidate_county_count": 2, "candidate_state_count": 1,
            })
        ):
            result = geospatial.geolocate_pixel_region("R1", [[0, 0], [1, 3]])
        self.assertEqual(result["region_scope"], "provided_region_pixel_centres")
        self.assertEqual(result["pixel_count"], 2)
        self.assertEqual({item["county"] for item in result["administrative_coverage"]}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
