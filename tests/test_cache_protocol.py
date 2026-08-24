from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from sevir_world.data_access import cache_layers


class CacheProtocolTests(unittest.TestCase):
    def test_cache_manifest_preserves_source_roles_and_no_answer_evidence(self) -> None:
        identity = {
            "request_key": "request-123",
            "canonical": {"event_id": "R19062708588472", "analysis_time_utc": "2019-06-27T08:58:20Z"},
        }
        stored: dict = {}

        def fake_atomic_json(_path: Path, payload: dict) -> None:
            stored.clear()
            stored.update(deepcopy(payload))

        def fake_read_container(_event_id: str, _layer: str):
            return deepcopy(stored) if stored else None

        with patch.object(cache_layers, "_atomic_json", fake_atomic_json), patch.object(
            cache_layers, "_read_container", fake_read_container
        ):
            cache_layers.write_manifest(
                "R19062708588472",
                "standardized",
                identity,
                source_role="era5_environment_analysis",
                validation={"target_timeseries_available": True},
                target_timeseries_file="era5_target_timeseries.json",
            )
            cache_layers.write_manifest(
                "R19062708588472",
                "standardized",
                identity,
                source_role="aurora_environment_forecast",
                validation={
                    "requested_leads_hours": [6, 12],
                    "completed_leads_hours": [6, 12],
                    "complete": True,
                },
                aurora_forecast_file="aurora_forecast.npz",
            )

            era5 = cache_layers.read_manifest(
                "R19062708588472",
                "standardized",
                source_role="era5_environment_analysis",
            )
            aurora = cache_layers.read_manifest(
                "R19062708588472",
                "standardized",
                source_role="aurora_environment_forecast",
            )
            status = cache_layers.layer_status("R19062708588472", identity)
            self.assertEqual(era5["source_role"], "era5_environment_analysis")
            self.assertEqual(aurora["source_role"], "aurora_environment_forecast")
            self.assertEqual(aurora["validation"]["completed_leads_hours"], [6, 12])
            self.assertEqual(
                status["layers"]["standardized"]["source_roles"],
                ["aurora_environment_forecast", "era5_environment_analysis"],
            )
            self.assertFalse(era5["contains_natural_language_answer"])
            def keys(value):
                if isinstance(value, dict):
                    return set(value) | set().union(*(keys(item) for item in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(item) for item in value)) if value else set()
                return set()

            self.assertTrue(
                {"answer", "final_answer", "natural_language_answer", "response_text"}.isdisjoint(
                    keys(stored)
                )
            )
            with self.assertRaisesRegex(ValueError, "answers cannot be stored"):
                cache_layers.write_manifest(
                    "R19062708588472",
                    "presentation",
                    identity,
                    source_role="presentation_artifact",
                    answer="预先写好的自然语言结论",
                )


if __name__ == "__main__":
    unittest.main()
