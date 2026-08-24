from __future__ import annotations

import threading
import time
from dataclasses import dataclass


TERMINAL_MODEL_ERROR_MARKERS = (
    "arrearage",
    "overdue-payment",
    "account is in good standing",
    "allocationquota.freetieronly",
    "free quota exhausted",
    "invalid api key",
    "invalidapikey",
    "authentication failed",
    "api returned http 401",
    "api returned http 403",
    "api返回http 401",
    "api返回http 403",
)


def is_terminal_model_error(error: BaseException) -> bool:
    """Classify authentication/account failures that must abort an Agent run."""
    if isinstance(error, ModelCircuitOpen):
        return bool(error.snapshot.reason_code)
    message = str(error).lower()
    return any(marker in message for marker in TERMINAL_MODEL_ERROR_MARKERS)


def terminal_model_error_code(error: BaseException) -> str | None:
    """Return a stable reason code for a terminal provider failure."""
    if isinstance(error, ModelCircuitOpen) and error.snapshot.reason_code:
        return error.snapshot.reason_code
    if not is_terminal_model_error(error):
        return None
    message = str(error).lower()
    if "arrearage" in message or "overdue-payment" in message:
        return "model_api_arrearage"
    if "allocationquota.freetieronly" in message or "free quota exhausted" in message:
        return "model_api_quota_exhausted"
    if any(marker in message for marker in (
        "invalid api key", "invalidapikey", "authentication failed",
        "api returned http 401", "api返回http 401",
    )):
        return "model_api_authentication_failed"
    return "model_api_terminal_failure"


@dataclass
class CircuitSnapshot:
    state: str
    failures: int
    opened_until: float
    reason_code: str | None = None


class ModelCircuitBreaker:
    """Small process-local breaker for one provider/model transport key."""

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 20.0) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown_seconds = max(0.01, float(cooldown_seconds))
        # record_failure() returns a snapshot while holding the lock.  RLock
        # avoids a self-deadlock on the first transport failure.
        self._lock = threading.RLock()
        self._failures: dict[str, int] = {}
        self._opened_until: dict[str, float] = {}
        self._reason_codes: dict[str, str] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            until = self._opened_until.get(key, 0.0)
            if until <= now:
                if until:
                    self._opened_until.pop(key, None)
                    self._failures[key] = 0
                    self._reason_codes.pop(key, None)
                return True
            return False

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._opened_until.pop(key, None)
            self._reason_codes.pop(key, None)

    def record_failure(self, key: str) -> CircuitSnapshot:
        now = time.monotonic()
        with self._lock:
            failures = self._failures.get(key, 0) + 1
            self._failures[key] = failures
            if failures >= self.threshold:
                self._opened_until[key] = now + self.cooldown_seconds
            return self.snapshot(key)

    def trip(
        self,
        key: str,
        cooldown_seconds: float | None = None,
        *,
        reason_code: str = "terminal_model_failure",
    ) -> CircuitSnapshot:
        """Open the circuit immediately after a non-retryable provider failure."""
        now = time.monotonic()
        cooldown = self.cooldown_seconds if cooldown_seconds is None else max(
            0.01, float(cooldown_seconds)
        )
        with self._lock:
            self._failures[key] = max(self.threshold, self._failures.get(key, 0))
            self._opened_until[key] = max(
                self._opened_until.get(key, 0.0),
                now + cooldown,
            )
            self._reason_codes[key] = str(reason_code)
            return self.snapshot(key)

    def snapshot(self, key: str) -> CircuitSnapshot:
        now = time.monotonic()
        with self._lock:
            until = self._opened_until.get(key, 0.0)
            return CircuitSnapshot(
                state="open" if until > now else "closed",
                failures=self._failures.get(key, 0),
                opened_until=until,
                reason_code=self._reason_codes.get(key),
            )

    def reset(self, key: str | None = None) -> None:
        """Reset one circuit (or every circuit) for tests and operator recovery."""
        with self._lock:
            if key is None:
                self._failures.clear()
                self._opened_until.clear()
                self._reason_codes.clear()
            else:
                self._failures.pop(key, None)
                self._opened_until.pop(key, None)
                self._reason_codes.pop(key, None)

    def snapshots(self) -> dict[str, CircuitSnapshot]:
        with self._lock:
            keys = set(self._failures) | set(self._opened_until)
        return {key: self.snapshot(key) for key in keys}


MODEL_CIRCUIT_BREAKER = ModelCircuitBreaker()


class ModelCircuitOpen(RuntimeError):
    def __init__(self, key: str, snapshot: CircuitSnapshot) -> None:
        self.key = key
        self.snapshot = snapshot
        reason = f"，原因：{snapshot.reason_code}" if snapshot.reason_code else ""
        super().__init__(
            f"模型传输熔断中：{key}{reason}，请在"
            f"{max(0, snapshot.opened_until - time.monotonic()):.1f}秒后重试"
        )
