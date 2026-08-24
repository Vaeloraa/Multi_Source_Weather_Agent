from __future__ import annotations

from pathlib import Path
import unittest

import sevir_world.tools as tools
from sevir_world.agent.contracts import CANONICAL_ARGUMENTS
from sevir_world.api.unified_service import bind_request_assets


ROOT = Path(__file__).resolve().parents[1]


class ToolCapabilityMigrationTests(unittest.TestCase):
    def test_agent_namespace_contains_only_four_capabilities_plus_ds(self) -> None:
        self.assertEqual(
            CANONICAL_ARGUMENTS,
            ("ds", "sevir", "geolocator", "cascast", "aurora"),
        )

    def test_legacy_implementations_are_retained_and_importable(self) -> None:
        legacy = {
            "CasCastForecaster": "forecast.py",
            "SevirHistory": "history.py",
            "SevirStatistics": "statistics.py",
            "UncertaintyAnalyzer": "uncertainty.py",
            "VerificationMetrics": "metrics.py",
            "Visualizer": "visualization.py",
            "FoundationModelContext": "foundation.py",
        }
        for export_name, filename in legacy.items():
            self.assertTrue(callable(getattr(tools, export_name)))
            self.assertTrue((ROOT / "sevir_world" / "tools" / filename).is_file())

    def test_new_capability_facades_are_importable(self) -> None:
        for export_name in (
            "SevirCapability", "GeolocatorCapability",
            "CasCastCapability", "AuroraCapability",
        ):
            self.assertTrue(callable(getattr(tools, export_name)))

    def test_legacy_client_tool_selection_is_migrated_at_ingress(self) -> None:
        payload = {
            "selected_tools": [
                "data", "history", "geolocator", "forecaster",
                "uncertainty", "foundation", "visualizer", "arrays",
            ]
        }
        _, selected = bind_request_assets("R19062708588472", payload)
        self.assertEqual(
            selected,
            ["ds", "sevir", "geolocator", "cascast", "aurora"],
        )


if __name__ == "__main__":
    unittest.main()
