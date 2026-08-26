from __future__ import annotations

"""OpenAI-compatible transport for one explicitly fixed model."""

import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sevir_world.api.model_router import (
    MODEL_AVAILABILITY_ROUTER,
    ModelCandidate,
    classify_model_failure,
)
from sevir_world.api.socks_transport import request_json_via_socks5


ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "models": os.getenv("DEEPSEEK_MODELS", "deepseek-chat,deepseek-reasoner"),
    },
    "kimi": {
        "label": "Kimi",
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "api_key": os.getenv("KIMI_API_KEY", ""),
        "model": os.getenv("KIMI_MODEL", "kimi-k2.5"),
        "models": os.getenv("KIMI_MODELS", ""),
    },
    "qwen": {
        "label": "Qwen",
        "base_url": os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "model": os.getenv("QWEN_MODEL", "qwen3.7-plus-2026-05-26"),
        "models": os.getenv("QWEN_MODELS", ""),
    },
}

FIXED_PROVIDER_NAME = os.getenv("MODEL_FIXED_PROVIDER", "qwen").strip() or "qwen"
FIXED_MODEL_NAME = os.getenv("MODEL_FIXED_NAME", "deepseek-v4-flash").strip() or "deepseek-v4-flash"


def fixed_model_selection(
    providers: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    """Return the only provider/model pair allowed for Agent requests."""

    registry = providers or PROVIDERS
    if FIXED_PROVIDER_NAME not in registry:
        raise RuntimeError(
            f"固定模型提供方未配置：{FIXED_PROVIDER_NAME}/{FIXED_MODEL_NAME}"
        )
    return FIXED_PROVIDER_NAME, FIXED_MODEL_NAME


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _endpoint(provider: dict[str, str]) -> str:
    return f"{provider['base_url'].rstrip('/')}/chat/completions"


def _call_once(
    candidate: ModelCandidate,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    provider = dict(candidate.provider)
    if not provider.get("api_key"):
        raise RuntimeError(f"{provider.get('label', candidate.provider_name)} API key未配置")
    payload: dict[str, Any] = {
        "model": candidate.model,
        "messages": messages,
        "temperature": 0.1,
        # The Reflective controller owns the execute-observe loop.  Stop the
        # text completion at the first action boundary so one model turn can
        # never emit a speculative chain of several execute blocks.
            "stop": ["</execute>", "</solution>", "</expand>"],
    }
    lowered = candidate.model.lower()
    if provider.get("label") == "Qwen":
        # The Agent protocol needs short action/answer messages.  An 8192
        # token ceiling lets a single reflective turn consume the whole
        # 30-second request budget even when the user asks a simple question.
        # Agent turns are either a compact Python action or a concise Chinese
        # synthesis.  A bounded output avoids spending the whole request
        # budget on speculative prose while leaving room for figures' URLs.
        payload["max_tokens"] = int(os.getenv("QWEN_MAX_TOKENS", "2048"))
        if lowered.startswith(("qwen", "qwq")):
            # Thinking output is decoded token by token and dominates request
            # latency.  Agent turns must stay compact action/answer messages,
            # so reasoning is opt-in instead of paying a 4096-token thinking
            # budget on every call.  Models without the parameter (e.g. the
            # fixed kimi-k2.6) deliberately receive none of it.
            payload["enable_thinking"] = _bool_env("QWEN_ENABLE_THINKING", False)
            payload["thinking_budget"] = int(os.getenv("QWEN_THINKING_BUDGET", "512"))
    url = _endpoint(provider)
    proxy = (
        os.getenv("QWEN_SOCKS5_PROXY", "").strip()
        if provider.get("label") == "Qwen"
        else ""
    )
    if proxy:
        result = request_json_via_socks5(
            "POST", url, provider["api_key"], proxy, timeout_seconds, payload
        )
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {provider['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            finally:
                error.close()
            raise RuntimeError(f"API returned HTTP {error.code}: {detail[:600]}") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as error:
            raise TimeoutError(f"model transport unavailable: {error}") from error
        if not raw:
            raise RuntimeError("model API returned an empty response")
        try:
            result = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as error:
            raise RuntimeError("model API returned invalid JSON") from error
    if not isinstance(result, dict) or not result.get("choices"):
        raise RuntimeError("model API response contains no choices")
    return result


def _candidate_sequence(
    provider: dict[str, str], model: str
) -> list[ModelCandidate]:
    name = MODEL_AVAILABILITY_ROUTER.provider_name(PROVIDERS, provider) or "selected"
    return [ModelCandidate(name, provider, model, "fixed")]


def post_chat_completion(
    provider: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Call the fixed model once; never retry with another model."""

    total = max(
        1.0,
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(os.getenv("MODEL_API_TIMEOUT_SECONDS", "120")),
    )
    candidate = _candidate_sequence(provider, str(model))[0]
    started_at = time.monotonic()
    try:
        result = _call_once(candidate, messages, total)
    except BaseException as error:
        decision = classify_model_failure(error)
        MODEL_AVAILABILITY_ROUTER.mark_failure(candidate, decision, PROVIDERS)
        raise RuntimeError(
            f"固定模型 {candidate.model} 调用失败；未尝试其他模型：{error}"
        ) from error
    MODEL_AVAILABILITY_ROUTER.mark_success(candidate, started_at=started_at)
    result["_sevir_model_route"] = {
        "provider": candidate.provider_name,
        "model": candidate.model,
        "failover_used": False,
        "attempts": [{
            "provider": candidate.provider_name,
            "model": candidate.model,
            "status": "selected",
            "reason": "fixed",
        }],
    }
    return result


def provider_status() -> dict[str, Any]:
    """Return the fixed model configuration, never API keys."""
    status: dict[str, Any] = {}
    for name, provider in PROVIDERS.items():
        enabled = name == FIXED_PROVIDER_NAME
        status[name] = {
            "label": provider["label"],
            "model": FIXED_MODEL_NAME if enabled else provider["model"],
            "enabled": enabled,
            "configured": bool(provider.get("api_key")),
            "configured_models": [FIXED_MODEL_NAME] if enabled else [],
            "discovered_models": [],
        }
    return status
