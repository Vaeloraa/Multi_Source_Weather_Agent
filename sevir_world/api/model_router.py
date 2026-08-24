"""OpenAI-compatible model discovery and bounded failover routing.

The router deliberately keeps transport policy separate from Agent reasoning.
It discovers model identifiers from ``GET /models`` (with a short cache),
remembers temporarily unusable provider/model pairs, and returns a bounded
fallback sequence.  It never stores or reports API keys.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .socks_transport import request_json_via_socks5


_NON_CHAT_MARKERS = (
    "embedding", "rerank", "text-to-image", "image", "audio", "speech",
    "tts", "asr", "whisper", "moderation", "ocr", "video",
)


def _csv_models(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = str(value or "").replace(";", ",").split(",")
    return list(dict.fromkeys(item.strip() for item in raw if item.strip()))


def _models_endpoint(base_url: str) -> str:
    return f"{str(base_url).rstrip('/')}/models"


def _looks_like_chat_model(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    return bool(lowered) and not any(marker in lowered for marker in _NON_CHAT_MARKERS)


@dataclass(frozen=True)
class ModelCandidate:
    provider_name: str
    provider: Mapping[str, Any]
    model: str
    source: str

    @property
    def key(self) -> str:
        return (
            f"{self.provider_name}:{self.model}:"
            f"{self.provider.get('base_url', '')}"
        )


@dataclass(frozen=True)
class FailureDecision:
    switchable: bool
    scope: str = "model"
    reason: str = "request_error"


def classify_model_failure(error: BaseException) -> FailureDecision:
    """Classify only service/model availability failures as failover-safe."""

    if isinstance(error, TimeoutError):
        return FailureDecision(True, "model", "model_transport_unavailable")
    message = str(error).lower()
    if any(marker in message for marker in (
        "arrearage", "overdue-payment", "invalid api key", "invalidapikey",
        "authentication failed", "api returned http 401", "api返回http 401",
    )):
        return FailureDecision(True, "provider", "provider_account_unavailable")
    if any(marker in message for marker in (
        "allocationquota.freetieronly", "free quota exhausted", "quota exceeded",
        "quota_exceeded", "quota_exhausted", "insufficient_quota", "resource_exhausted",
    )):
        return FailureDecision(True, "model", "model_quota_exhausted")
    if any(marker in message for marker in (
        "rate limit", "rate_limit", "too many requests", "http 429",
        "http 403", "http 404", "model not found", "model_not_found",
        "does not exist", "not have access to model",
    )):
        return FailureDecision(True, "model", "model_unavailable")
    if any(marker in message for marker in (
        "http 500", "http 502", "http 503", "http 504", "service unavailable",
        "temporarily unavailable", "overloaded", "timed out", "timeout",
        "connection reset", "connection aborted", "remote disconnected",
        "cannot connect", "unable to connect", "无法连接模型api",
        "模型api在", "模型api调用已达到", "总时间上限", "模型api返回空响应", "非法 json",
        "模型传输熔断中", "model circuit open", "circuit breaker open",
    )):
        return FailureDecision(True, "model", "model_transport_unavailable")
    return FailureDecision(False)


class ModelAvailabilityRouter:
    """Process-local discovery cache and availability memory."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._discoveries: dict[str, tuple[float, tuple[str, ...]]] = {}
        self._unavailable_until: dict[str, float] = {}
        self._unavailable_since: dict[str, float] = {}
        self._provider_unavailable_until: dict[str, float] = {}
        self._provider_unavailable_since: dict[str, float] = {}
        self._inflight: dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self._discoveries.clear()
            self._unavailable_until.clear()
            self._unavailable_since.clear()
            self._provider_unavailable_until.clear()
            self._provider_unavailable_since.clear()
            self._inflight.clear()

    @staticmethod
    def provider_name(
        providers: Mapping[str, Mapping[str, Any]], provider: Mapping[str, Any]
    ) -> str | None:
        for name, configured in providers.items():
            if configured is provider:
                return name
            if (
                configured.get("base_url") == provider.get("base_url")
                and configured.get("label") == provider.get("label")
            ):
                return name
        return None

    def discover_models(
        self,
        provider_name: str,
        provider: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> list[str]:
        """Return chat-capable IDs advertised by an OpenAI-compatible API."""

        if not provider.get("api_key") or not provider.get("base_url"):
            return []
        with self._lock:
            if self._provider_unavailable_until.get(provider_name, 0.0) > time.monotonic():
                return []
        cache_key = f"{provider_name}:{provider.get('base_url', '')}"
        now = time.monotonic()
        ttl = max(5.0, float(os.getenv("MODEL_DISCOVERY_TTL_SECONDS", "300")))
        with self._lock:
            cached = self._discoveries.get(cache_key)
            if cached and not force and cached[0] > now:
                return list(cached[1])

        endpoint = _models_endpoint(str(provider["base_url"]))
        request = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {provider['api_key']}",
                "Accept": "application/json",
            },
            method="GET",
        )
        timeout = max(
            0.2, min(10.0, float(os.getenv("MODEL_DISCOVERY_TIMEOUT_SECONDS", "2")))
        )
        models: list[str] = []
        try:
            socks5_proxy = (
                os.getenv("QWEN_SOCKS5_PROXY", "").strip()
                if provider.get("label") == "Qwen"
                else ""
            )
            if socks5_proxy:
                payload = request_json_via_socks5(
                    "GET", endpoint, str(provider["api_key"]), socks5_proxy, timeout
                )
            else:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            for row in rows:
                model_id = str(row.get("id", "")).strip() if isinstance(row, dict) else ""
                if _looks_like_chat_model(model_id):
                    models.append(model_id)
        except (OSError, RuntimeError, ValueError, TypeError, urllib.error.URLError):
            # Discovery is an optimisation. Configured models remain usable
            # when a compatible gateway does not expose GET /models.
            models = []
        models = list(dict.fromkeys(models))
        with self._lock:
            self._discoveries[cache_key] = (now + ttl, tuple(models))
        return models

    def _configured_models(self, provider: Mapping[str, Any]) -> list[str]:
        return list(dict.fromkeys(
            _csv_models(provider.get("model")) + _csv_models(provider.get("models"))
        ))

    def fallback_candidates(
        self,
        providers: Mapping[str, Mapping[str, Any]],
        primary_provider: Mapping[str, Any],
        primary_model: str,
    ) -> list[ModelCandidate]:
        """Build a deterministic, deduplicated, bounded fallback sequence."""

        primary_name = self.provider_name(providers, primary_provider)
        if primary_name is None:
            return []
        configured_order = _csv_models(os.getenv("MODEL_PROVIDER_FALLBACK_ORDER", ""))
        names = [primary_name]
        names.extend(name for name in configured_order if name in providers and name not in names)
        names.extend(name for name in providers if name not in names)

        candidates: list[ModelCandidate] = []
        seen = {(primary_name, str(primary_model))}
        for name in names:
            provider = providers[name]
            if not provider.get("api_key"):
                continue
            with self._lock:
                if self._provider_unavailable_until.get(name, 0.0) > time.monotonic():
                    continue
            configured = self._configured_models(provider)
            discovered = self.discover_models(name, provider)
            # The API-advertised order is not assumed to imply quality. Keep
            # operator-configured preferences first, then advertised models.
            for model, source in (
                [(item, "configured") for item in configured]
                + [(item, "discovered") for item in discovered]
            ):
                identity = (name, model)
                if identity in seen:
                    continue
                seen.add(identity)
                candidate = ModelCandidate(name, provider, model, source)
                if self.is_available(candidate.key):
                    candidates.append(candidate)
        limit = max(1, min(12, int(os.getenv("MODEL_FAILOVER_MAX_CANDIDATES", "4"))))
        return candidates[:limit]

    def is_available(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            until = self._unavailable_until.get(key, 0.0)
            if until <= now:
                self._unavailable_until.pop(key, None)
                return True
            return False

    def mark_failure(
        self,
        candidate: ModelCandidate,
        decision: FailureDecision,
        providers: Mapping[str, Mapping[str, Any]],
    ) -> None:
        cooldown_name = (
            "MODEL_TRANSPORT_COOLDOWN_SECONDS"
            if decision.reason == "model_transport_unavailable"
            else "MODEL_FAILOVER_COOLDOWN_SECONDS"
        )
        cooldown_default = "30" if cooldown_name == "MODEL_TRANSPORT_COOLDOWN_SECONDS" else "900"
        cooldown = max(1.0, float(os.getenv(cooldown_name, cooldown_default)))
        failed_at = time.monotonic()
        until = failed_at + cooldown
        with self._lock:
            if decision.scope == "provider":
                self._provider_unavailable_until[candidate.provider_name] = until
                self._provider_unavailable_since[candidate.provider_name] = failed_at
                for model in self._configured_models(candidate.provider):
                    key = ModelCandidate(
                        candidate.provider_name, candidate.provider, model, "configured"
                    ).key
                    self._unavailable_until[key] = until
                    self._unavailable_since[key] = failed_at
                # Also suppress every currently discovered model for the
                # account; an authentication/account error is not model-local.
                # Do not issue another network request with known-bad account
                # credentials merely to populate this list.
                discovery_key = (
                    f"{candidate.provider_name}:"
                    f"{candidate.provider.get('base_url', '')}"
                )
                cached_models = self._discoveries.get(discovery_key, (0.0, ()))[1]
                for model in cached_models:
                    key = ModelCandidate(
                        candidate.provider_name, candidate.provider, model, "discovered"
                    ).key
                    self._unavailable_until[key] = until
                    self._unavailable_since[key] = failed_at
            else:
                self._unavailable_until[candidate.key] = until
                self._unavailable_since[candidate.key] = failed_at

    def mark_success(self, candidate: ModelCandidate, *, started_at: float | None = None) -> None:
        with self._lock:
            model_failure = self._unavailable_since.get(candidate.key, 0.0)
            provider_failure = self._provider_unavailable_since.get(candidate.provider_name, 0.0)
            # A call that began before a concurrent failure is stale evidence:
            # its late success must not erase the newer cooldown decision.
            if started_at is None or model_failure <= started_at:
                self._unavailable_until.pop(candidate.key, None)
                self._unavailable_since.pop(candidate.key, None)
            if started_at is None or provider_failure <= started_at:
                self._provider_unavailable_until.pop(candidate.provider_name, None)
                self._provider_unavailable_since.pop(candidate.provider_name, None)

    def claim_if_idle(self, candidate: ModelCandidate) -> float | None:
        """Atomically reserve an available candidate only when it has no work."""
        with self._lock:
            now = time.monotonic()
            if self._provider_unavailable_until.get(candidate.provider_name, 0.0) > now:
                return None
            if self._unavailable_until.get(candidate.key, 0.0) > now:
                return None
            if self._inflight.get(candidate.key, 0) > 0:
                return None
            self._inflight[candidate.key] = 1
            return now

    def claim_least_busy(
        self, candidates: list[ModelCandidate]
    ) -> tuple[ModelCandidate, float] | None:
        """Reserve the least-loaded available model, preserving preference ties."""
        with self._lock:
            now = time.monotonic()
            available = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if self._provider_unavailable_until.get(candidate.provider_name, 0.0) <= now
                and self._unavailable_until.get(candidate.key, 0.0) <= now
            ]
            if not available:
                return None
            _, selected = min(
                available,
                key=lambda item: (self._inflight.get(item[1].key, 0), item[0]),
            )
            self._inflight[selected.key] = self._inflight.get(selected.key, 0) + 1
            return selected, now

    def release(self, candidate: ModelCandidate) -> None:
        with self._lock:
            remaining = self._inflight.get(candidate.key, 0) - 1
            if remaining > 0:
                self._inflight[candidate.key] = remaining
            else:
                self._inflight.pop(candidate.key, None)


MODEL_AVAILABILITY_ROUTER = ModelAvailabilityRouter()
