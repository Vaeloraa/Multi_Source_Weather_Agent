"""Application bootstrap and shared Agent/model settings.

Tool-specific settings live with each tool; mutable storage state belongs to
the persistence layer. Importing this module must not initialize either.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # The project-root .env is the sole production configuration source.
        # Tests may still patch os.environ after import through dependency
        # injection, but process-launch environment does not silently create a
        # second production configuration layer.
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_dotenv()

VENDOR = ROOT / "vendor"
# The bundled vendor wheel is a portability fallback.  Prefer the active
# environment's compiled packages (notably h5py) unless explicitly forced;
# the vendored h5py binary can be incompatible with the current Python.
if VENDOR.exists() and os.getenv("SEVIR_USE_VENDOR", "0").lower() in {"1", "true", "yes"}:
    sys.path.insert(0, str(VENDOR))


def _bool_env(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raise RuntimeError(f"Configure {name} in .env")
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} in .env must be a boolean")


def _float_env(name: str, minimum: float = 0.0) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        raise RuntimeError(f"Configure {name} in .env")
    try:
        value = float(raw)
        if not math.isfinite(value) or value < minimum:
            raise ValueError(name)
        return value
    except ValueError as error:
        raise RuntimeError(f"{name} in .env must be numeric") from error


def _int_env(name: str, minimum: int = 1) -> int:
    value = _float_env(name, minimum)
    if not value.is_integer():
        raise RuntimeError(f"{name} in .env must be an integer")
    return int(value)


def model_generation_settings() -> dict[str, Any]:
    """Return the single .env-backed generation profile for every Agent call."""

    return {
        "temperature": _float_env("MODEL_TEMPERATURE"),
        "max_tokens": _int_env("MODEL_MAX_TOKENS"),
        "enable_thinking": _bool_env("MODEL_ENABLE_THINKING"),
        "thinking_budget": _int_env("MODEL_THINKING_BUDGET", 0),
    }


def model_api_timeout_seconds() -> float:
    """Return the one per-call transport timeout configured in .env."""

    return _float_env("MODEL_API_TIMEOUT_SECONDS", 0.001)


def configured_context_settings() -> dict[str, int]:
    """One .env-backed context policy; output reserve matches model generation."""
    return {
        "window_tokens": _int_env("MODEL_CONTEXT_WINDOW_TOKENS"),
        "input_tokens": _int_env("SEVIR_CONTEXT_INPUT_TOKENS"),
        "target_tokens": _int_env("SEVIR_CONTEXT_TARGET_TOKENS"),
        "output_tokens": _int_env("MODEL_MAX_TOKENS"),
        "margin_tokens": _int_env("SEVIR_CONTEXT_MARGIN_TOKENS"),
        "summary_tokens": _int_env("SEVIR_CONTEXT_SUMMARY_TOKENS"),
        "recent_turns": _int_env("SEVIR_CONTEXT_RECENT_TURNS", 0),
        "result_tokens": _int_env("SEVIR_CONTEXT_RESULT_TOKENS"),
    }


def judge_timeout_seconds() -> float:
    return _float_env("SEVIR_JUDGE_TIMEOUT_SECONDS", 0.001)


def configured_agent_request_budget_seconds(data_policy: str | None = None) -> float:
    name = (
        "SEVIR_LIVE_AGENT_REQUEST_TIMEOUT_SECONDS"
        if str(data_policy or "").strip().lower() == "live"
        else "SEVIR_AGENT_REQUEST_TIMEOUT_SECONDS"
    )
    return _float_env(name, 0.001)


def configured_agent_limits() -> dict[str, int]:
    return {
        "max_steps": _int_env("SEVIR_UNIFIED_MAX_STEPS"),
        "max_executions": _int_env("SEVIR_UNIFIED_MAX_EXECUTIONS"),
        "max_error_retries": _int_env("SEVIR_UNIFIED_MAX_ERROR_RETRIES"),
    }


def configured_model_concurrency() -> int:
    return _int_env("SEVIR_MODEL_CONCURRENCY")


def configured_agent_execution_timeout_seconds() -> float:
    return _float_env("SEVIR_AGENT_EXECUTION_TIMEOUT_SECONDS", 0.001)
