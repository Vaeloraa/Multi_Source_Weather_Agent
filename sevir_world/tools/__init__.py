"""Public resources injected as independent ``run()`` arguments.

The exports are lazy so importing one tool does not initialize all optional
model/data stacks in a worker process.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SevirCapability",
    "GeolocatorCapability",
    "CasCastCapability",
    "AuroraCapability",
    "CasCastForecaster",
    "SevirGeolocator",
    "SevirHistory",
    "SevirStatistics",
    "UncertaintyAnalyzer",
    "VerificationMetrics",
    "Visualizer",
    "FoundationModelContext",
]

_EXPORTS = {
    "SevirCapability": ("capabilities", "SevirCapability"),
    "GeolocatorCapability": ("capabilities", "GeolocatorCapability"),
    "CasCastCapability": ("capabilities", "CasCastCapability"),
    "AuroraCapability": ("capabilities", "AuroraCapability"),
    "CasCastForecaster": ("forecast", "CasCastForecaster"),
    "SevirGeolocator": ("geolocation", "SevirGeolocator"),
    "SevirHistory": ("history", "SevirHistory"),
    "VerificationMetrics": ("metrics", "VerificationMetrics"),
    "SevirStatistics": ("statistics", "SevirStatistics"),
    "UncertaintyAnalyzer": ("uncertainty", "UncertaintyAnalyzer"),
    "Visualizer": ("visualization", "Visualizer"),
    "FoundationModelContext": ("foundation", "FoundationModelContext"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}.{module_name}"), attribute)
