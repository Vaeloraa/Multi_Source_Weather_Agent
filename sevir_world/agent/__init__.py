"""LLM-facing request, prompt, and Agent orchestration (lazy exports)."""
from __future__ import annotations

from typing import Any

__all__ = ["DataIndexer", "ExecutionEnvironment", "ExecutionRequest", "run_unified_agent"]


def __getattr__(name: str) -> Any:
    if name == "ExecutionRequest":
        from .contracts import ExecutionRequest
        return ExecutionRequest
    if name in {"DataIndexer", "ExecutionEnvironment"}:
        from .runtime import DataIndexer, ExecutionEnvironment
        return {"DataIndexer": DataIndexer, "ExecutionEnvironment": ExecutionEnvironment}[name]
    if name == "run_unified_agent":
        from .unified import run_unified_agent
        return run_unified_agent
    raise AttributeError(name)
