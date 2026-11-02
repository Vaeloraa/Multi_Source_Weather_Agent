"""Model selection, chat payloads and response validation for the application."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping
from weather_agent.config import (
    model_api_timeout_seconds, model_generation_settings, configured_context_settings,
)
from weather_agent.socks_transport import request_via_socks5
from weather_agent.errors import ClassifiedError, FailureClass, classify_failure


PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", ""),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
    "kimi": {
        "label": "Kimi",
        "base_url": os.getenv("KIMI_BASE_URL", ""),
        "api_key": os.getenv("KIMI_API_KEY", ""),
    },
    "qwen": {
        "label": "Qwen",
        "base_url": os.getenv("QWEN_BASE_URL", ""),
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    },
}

FIXED_PROVIDER_NAME = os.getenv("MODEL_FIXED_PROVIDER", "").strip()
FIXED_MODEL_NAME = os.getenv("MODEL_FIXED_NAME", "").strip()


def fixed_model_selection(
    providers: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Return the only provider/model pair allowed for Agent requests."""

    registry = providers or PROVIDERS
    missing = [
        name
        for name, value in (
            ("MODEL_FIXED_PROVIDER", FIXED_PROVIDER_NAME),
            ("MODEL_FIXED_NAME", FIXED_MODEL_NAME),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Configure the fixed model in .env: " + ", ".join(missing)
        )
    if FIXED_PROVIDER_NAME not in registry:
        raise RuntimeError(
            f"Fixed model provider is not configured: {FIXED_PROVIDER_NAME}/{FIXED_MODEL_NAME}"
        )
    return FIXED_PROVIDER_NAME, FIXED_MODEL_NAME


def judge_model_selection() -> tuple[str, str]:
    """Return the independent Judge identity from the same .env file."""

    provider = str(os.getenv("WEATHER_AGENT_JUDGE_PROVIDER", "")).strip()
    model = str(os.getenv("WEATHER_AGENT_JUDGE_MODEL", "")).strip()
    if not provider or not model:
        raise RuntimeError("Configure WEATHER_AGENT_JUDGE_PROVIDER and WEATHER_AGENT_JUDGE_MODEL in .env")
    if provider not in PROVIDERS:
        raise RuntimeError(f"Judge model provider is not configured: {provider}/{model}")
    return provider, model


def _decode_response(status: int, raw: bytes) -> dict[str, Any]:
    """Validate direct and proxied model responses through the same path."""
    text = raw.decode("utf-8", errors="replace")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        if 200 <= status < 300:
            raise ClassifiedError(
                "model API returned empty or invalid JSON", FailureClass.MODEL_PROTOCOL_ERROR,
            ) from error
        result = {}
    if not 200 <= status < 300:
        details = result.get("error") if isinstance(result, dict) else None
        record = dict(details) if isinstance(details, dict) else {}
        record["http_status"] = status
        error = ClassifiedError(f"API returned HTTP {status}: {text[:600]}", classify_failure(record))
        error.http_status = status
        error.code = record.get("code")
        raise error
    if not isinstance(result, dict) or not result.get("choices"):
        raise ClassifiedError("model API response contains no choices", FailureClass.MODEL_PROTOCOL_ERROR)
    return result


def _call_once(
    provider_name: str,
    provider_value: Mapping[str, str],
    model: str,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
    *, summary: bool = False,
) -> dict[str, Any]:
    provider = dict(provider_value)
    if not provider.get("api_key"):
        raise RuntimeError(f"{provider.get('label', provider_name)} API key is not configured")
    if not provider.get("base_url"):
        raise RuntimeError(f"{provider.get('label', provider_name)} base URL is not configured")
    lowered = model.lower()
    generation = model_generation_settings()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": generation["temperature"],
        "max_tokens": configured_context_settings()["summary_tokens"] if summary else generation["max_tokens"],
    }
    if not summary:
        payload["stop"] = ["</execute>", "</solution>", "</read_context>", "</select_tools>"]
    if provider.get("label") == "Qwen":
        # DashScope serves multiple model families with different thinking flags.
        if lowered != "kimi-k2-thinking":
            payload["enable_thinking"] = False if summary else generation["enable_thinking"]
        if not summary and lowered.startswith(("qwen", "qwq")):
            payload["thinking_budget"] = generation["thinking_budget"]
    url = f"{provider['base_url'].rstrip('/')}/chat/completions"
    proxy = (
        os.getenv("QWEN_SOCKS5_PROXY", "").strip()
        if provider.get("label") == "Qwen"
        else ""
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if proxy:
        status, raw = request_via_socks5(
            "POST", url, headers, proxy, timeout_seconds, body,
        )
    else:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                status, raw = error.code, error.read()
            finally:
                error.close()
    return _decode_response(status, raw)


def _provider_name(provider: Mapping[str, str]) -> str:
    """Return the configured name of the one explicitly selected provider."""

    for name, configured in PROVIDERS.items():
        if configured is provider or (
            configured.get("base_url") == provider.get("base_url")
            and configured.get("label") == provider.get("label")
        ):
            return name
    return "selected"


def post_chat_completion(
    provider: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Call the fixed model once; never retry with another model."""

    total = model_api_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    provider_name = _provider_name(provider)
    result = _call_once(provider_name, provider, str(model), messages, total)
    result["_weather_agent_model_route"] = {
        "provider": provider_name,
        "model": str(model),
        "failover_used": False,
        "attempts": [{
            "provider": provider_name,
            "model": str(model),
            "status": "selected",
            "reason": "fixed",
        }],
    }
    return result


def post_context_summary(
    provider: dict[str, str], model: str, messages: list[dict[str, Any]],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Same fixed provider/transport; a bounded JSON summary without action stops."""
    timeout = model_api_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    return _call_once(_provider_name(provider), provider, model, messages, timeout, summary=True)


def provider_status() -> dict[str, Any]:
    """Return the fixed model configuration, never API keys."""
    status: dict[str, Any] = {}
    for name, provider in PROVIDERS.items():
        enabled = name == FIXED_PROVIDER_NAME
        status[name] = {
            "label": provider["label"],
            "model": FIXED_MODEL_NAME if enabled else None,
            "enabled": enabled,
            "configured": bool(provider.get("api_key") and provider.get("base_url")),
            "configured_models": [FIXED_MODEL_NAME] if enabled else [],
            "discovered_models": [],
        }
    return status
