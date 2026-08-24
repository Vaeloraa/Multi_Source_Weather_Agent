from __future__ import annotations

"""Descriptive model metadata; never an Agent plan or method allow-list."""

from typing import Any


_CAPABILITIES: dict[str, dict[str, Any]] = {
    "sevir": {
        "model": "sevir",
        "inputs": ["event identity or uploaded grid"],
        "outputs": ["encoded VIL observation"],
        "scales": ["local"],
        "tasks": ["observation"],
        "forecast_leads_hours": [],
    },
    "cascast": {
        "model": "cascast",
        "inputs": ["13 consecutive local observation frames"],
        "outputs": ["ensemble encoded-VIL forecast grids"],
        "scales": ["local"],
        "tasks": ["forecast", "verification"],
        "forecast_leads_hours": [1, 2, 3],
    },
    "era5": {
        "model": "era5",
        "inputs": ["time", "location or region"],
        "outputs": ["environmental fields"],
        "scales": ["regional", "large_scale"],
        "tasks": ["reanalysis", "observation"],
        "forecast_leads_hours": [],
    },
    "aurora": {
        "model": "aurora",
        "inputs": ["ERA5-compatible initial state"],
        "outputs": ["forecast environmental fields"],
        "scales": ["regional", "large_scale"],
        "tasks": ["forecast", "verification"],
        "forecast_leads_hours": [6, 12, 18, 24],
    },
}


def capability_snapshot() -> dict[str, Any]:
    return {
        name: {
            key: list(value) if isinstance(value, list) else value
            for key, value in record.items()
        }
        for name, record in _CAPABILITIES.items()
    }
