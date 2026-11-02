"""Public APIs for Multi-Source Weather Agent.

Public conveniences are resolved lazily.  Importing a narrow submodule such as
``weather_agent.code_execution.safety`` must not initialize every weather model,
plotting backend and data adapter inside an isolated worker.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Optional bundled dependencies must be available before submodule imports in
# both the web service and isolated workers.
_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

__all__ = ["Aurora", "CasCast", "Geospatial", "Sevir", "RequestBinding"]


def __getattr__(name: str) -> Any:
    if name in {"Aurora", "CasCast", "Geospatial", "Sevir"}:
        from .capabilities import Aurora, CasCast, Geospatial, Sevir
        return {"Aurora": Aurora, "CasCast": CasCast, "Geospatial": Geospatial, "Sevir": Sevir}[name]
    if name == "RequestBinding":
        from .capabilities.base import RequestBinding
        return RequestBinding
    raise AttributeError(name)
