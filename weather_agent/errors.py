from __future__ import annotations

from enum import StrEnum
import errno
import http.client
import ssl
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError


class FailureClass(StrEnum):
    """Stable failure codes with their public explanation and recovery action."""

    guidance: tuple[str, str]

    def __new__(cls, code: str, message: str, next_action: str) -> FailureClass:
        member = str.__new__(cls, code)
        member._value_ = code
        member.guidance = (message, next_action)
        return member

    PROVIDER_QUOTA_EXHAUSTED = ('provider_quota_exhausted', 'The model provider quota has been exhausted.', 'Add quota or select an available provider, then retry. Completed data retrieval remains available.')
    PROVIDER_MODEL_UNAVAILABLE = ('provider_model_unavailable', 'The configured model is unavailable from this provider.', 'Choose a model listed by /v1/models and update the model name in .env.')
    MODEL_PARAMETER_ERROR = ('model_parameter_error', 'The model does not support a request parameter.', 'Remove unsupported parameters. Enable thinking options only for models that support them.')
    PROVIDER_AUTHENTICATION = ('provider_authentication', 'The model provider rejected the credentials.', 'Check the API key, endpoint and provider account.')
    TRANSPORT_TLS_ERROR = ('transport_tls_error', 'The HTTPS connection ended unexpectedly.', 'Check the proxy, certificates and gateway, then retry.')
    LOCAL_ASSET_MISSING = ('local_asset_missing', 'A required local file or capability module is missing.', 'Check the reported path and restore the file or update the path configuration.')
    CACHE_MISS = ('cache_miss', 'No cached result matches the event, time and inputs.', 'Retrieve data or run inference if allowed, then cache the result under the same key.')
    TRANSPORT_CIRCUIT_OPEN = ('transport_circuit_open', 'Repeated connection failures have temporarily paused requests.', 'Wait for the retry window or restore the upstream service.')
    PROVIDER_RATE_LIMIT = ('provider_rate_limit', 'The model provider is busy or rate limited.', 'Retry later or use another available provider.')
    PROVIDER_5XX = ('provider_5xx', 'The model service encountered a temporary error.', 'Keep completed steps and retry the failed step later.')
    TRANSPORT_CONNECT_TIMEOUT = ('transport_connect_timeout', 'The connection to a model or data service timed out.', 'Check the network and service endpoint, then retry this step.')
    EXECUTION_TIMEOUT = ('execution_timeout', 'Analysis or inference exceeded the execution time limit.', 'Inspect the stalled step, reduce the analysis scope or reuse a cached result.')
    TRANSPORT_READ_TIMEOUT = ('transport_read_timeout', 'The remote service did not respond within the time limit.', 'Check the service. For active inference, wait or stop the task and retry.')
    TRANSPORT_RESET = ('transport_reset', 'The remote connection was interrupted.', 'Check the model gateway and retry. Completed tool results remain available.')
    ANALYSIS_BUDGET_EXHAUSTED = ('analysis_budget_exhausted', 'The analysis step budget has been reached.', 'Use the available results or increase the budget and resume the unfinished work.')
    CONTEXT_BUDGET_EXCEEDED = ('context_budget_exceeded', 'The question and required tool instructions exceed the input budget.', 'Narrow the question or tool selection, or adjust the budget to the model capacity.')
    RENDER_ERROR = ('render_error', 'Chart generation or validation failed.', 'Keep the numerical results and check the chart inputs, empty arrays and output path.')
    TOOL_ARGUMENT_ERROR = ('tool_argument_error', 'A tool received invalid arguments.', 'Check the tool name and arguments, then rerun the corrected step.')
    TOOL_RUNTIME_ERROR = ('tool_runtime_error', 'A tool failed during execution.', 'Expand the failed step for details and reuse successful upstream results.')
    CLIENT_DISCONNECTED = ('client_disconnected', 'The browser disconnected and the request was cancelled.', 'Reconnect and submit again. Matching cached results can be reused.')
    MODEL_PROTOCOL_ERROR = ('model_protocol_error', 'The model returned an incomplete or invalid action.', 'Retry the model step or choose a model that follows the action format reliably.')
    # Legacy task records only; the current Agent does not run patch repair.
    CODE_REPAIR_FAILED = ('code_repair_failed', 'A legacy repair attempt did not complete.', 'Review the original error and continue with a new code execution step.')
    MODEL_TRANSPORT_FAILURE = ('model_transport_failure', 'The language model request failed before returning a complete response.', 'Inspect the model error and reuse completed tool results when retrying.')
    SOLUTION_EVIDENCE_ERROR = ('solution_evidence_error', 'The answer is missing required supporting results.', 'Complete the missing observation, forecast or historical analysis, then revise the answer.')
    MISSING_REQUIRED_VISUAL = ('missing_required_visual', 'No valid new chart was generated for this turn.', 'Generate the requested chart using existing data, or turn off Generate charts and resubmit.')
    CONSECUTIVE_EXECUTION_ERRORS = ('consecutive_execution_errors', 'Repeated code or tool errors stopped the analysis.', 'Review the recent steps and fix the first underlying tool error before continuing.')
    SERVICE_RESTARTED = ('service_restarted', 'The service restarted while the task was running.', 'Resubmit the question. Saved sessions, snapshots and matching cached results remain available.')
    ANALYSIS_INCOMPLETE = ('analysis_incomplete', 'The requested analysis is incomplete.', 'Narrow the scope or provide the missing data.')
    UNKNOWN_ERROR = ('unknown_error', 'An unexpected error occurred.', 'Expand the failed step to inspect the original error and decide whether to retry.')


class ClassifiedError(RuntimeError):
    """An error whose public failure category is known at the throw site."""

    def __init__(self, message: str, failure_class: FailureClass | str) -> None:
        super().__init__(message)
        self.failure_class = str(failure_class)


_TERMINATION_FAILURE_CLASSES = {
    **{item.value: item.value for item in FailureClass},
    "client_cancelled": FailureClass.CLIENT_DISCONNECTED.value,
    "max_steps_without_solution": FailureClass.ANALYSIS_BUDGET_EXHAUSTED.value,
    "tool_cost_budget_exhausted": FailureClass.ANALYSIS_BUDGET_EXHAUSTED.value,
    "model_failure_after_observation": FailureClass.MODEL_TRANSPORT_FAILURE.value,
    "repair_scope_violation": FailureClass.CODE_REPAIR_FAILED.value,
}


def failure_class_for_termination(reason: str | None) -> str | None:
    """Map a known machine code without guessing from diagnostic prose."""

    return _TERMINATION_FAILURE_CLASSES.get(str(reason or "").strip())


_ERROR_CODES = {
    "ARGUMENT_INVALID": FailureClass.TOOL_ARGUMENT_ERROR,
    "CACHE_ONLY_MISS": FailureClass.CACHE_MISS,
    "insufficient_quota": FailureClass.PROVIDER_QUOTA_EXHAUSTED,
    "AllocationQuota.FreeTierOnly": FailureClass.PROVIDER_QUOTA_EXHAUSTED,
    "model_not_found": FailureClass.PROVIDER_MODEL_UNAVAILABLE,
    "unsupported_parameter": FailureClass.MODEL_PARAMETER_ERROR,
    "invalid_api_key": FailureClass.PROVIDER_AUTHENTICATION,
}
_ERROR_TYPES = {
    "ToolCostBudgetExceeded": FailureClass.ANALYSIS_BUDGET_EXHAUSTED,
    "ExecutionTimeout": FailureClass.EXECUTION_TIMEOUT,
    "ExecutionCancelled": FailureClass.CLIENT_DISCONNECTED,
    "WorkerProtocolError": FailureClass.MODEL_PROTOCOL_ERROR,
}
_TRANSIENT_ERRNOS = {
    errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
    errno.EPIPE, errno.ENETUNREACH, errno.EHOSTUNREACH, 10053, 10054, 10060, 10061,
}


def is_transient_transport_error(error: BaseException) -> bool:
    """Allow bounded retries only for identified transport failures."""
    if isinstance(error, HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    if isinstance(error, ssl.SSLError):
        return False
    if isinstance(error, (TimeoutError, ConnectionError, http.client.RemoteDisconnected)):
        return True
    if isinstance(error, URLError) and isinstance(error.reason, BaseException):
        return is_transient_transport_error(error.reason)
    return isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS


def classify_failure(error: BaseException | str | Mapping[str, Any], stage: str = "", _depth: int = 0) -> str:
    """Classify declared codes and typed errors; diagnostic prose is never a rule."""
    if _depth > 8:
        return FailureClass.UNKNOWN_ERROR
    record = error if isinstance(error, Mapping) else {}
    declared = record.get("failure_class") or getattr(error, "failure_class", None)
    if declared and str(declared) != FailureClass.UNKNOWN_ERROR:
        return str(declared)
    code = record.get("code") or getattr(error, "code", None)
    known = _ERROR_CODES.get(str(code)) or failure_class_for_termination(str(code or ""))
    if known:
        return known
    if isinstance(error, str):
        return _ERROR_CODES.get(error) or failure_class_for_termination(error) or FailureClass.UNKNOWN_ERROR
    error_type = record.get("type") or type(error).__name__
    if error_type in _ERROR_TYPES:
        return _ERROR_TYPES[error_type]
    status = record.get("http_status") or getattr(error, "http_status", None)
    if isinstance(error, HTTPError):
        status = error.code
    if status in {401, 403}:
        return FailureClass.PROVIDER_AUTHENTICATION
    if status == 429:
        return FailureClass.PROVIDER_RATE_LIMIT
    if isinstance(status, int) and 500 <= status < 600:
        return FailureClass.PROVIDER_5XX
    if isinstance(error, ssl.SSLError):
        return FailureClass.TRANSPORT_TLS_ERROR
    if isinstance(error, FileNotFoundError):
        return FailureClass.LOCAL_ASSET_MISSING
    if isinstance(error, TimeoutError):
        return FailureClass.EXECUTION_TIMEOUT if stage in {"execution", "worker"} else FailureClass.TRANSPORT_READ_TIMEOUT
    if isinstance(error, (ConnectionError, http.client.RemoteDisconnected)):
        return FailureClass.TRANSPORT_RESET
    cause = record.get("cause") or record.get("error")
    if isinstance(error, BaseException):
        cause = error.__cause__
        if isinstance(error, URLError):
            cause = error.reason
    if isinstance(cause, (BaseException, Mapping)):
        return classify_failure(cause, stage, _depth + 1)
    return FailureClass.UNKNOWN_ERROR


def error_record(error: BaseException, *, stage: str = "", _depth: int = 0) -> dict[str, Any]:
    """Preserve structured diagnostics once at serialization boundaries."""
    record: dict[str, Any] = {
        "type": type(error).__name__, "message": str(error),
        "failure_class": classify_failure(error, stage),
    }
    for name in ("code", "phase", "details", "issues", "resource_id", "backend_executed", "retryable", "http_status"):
        value = getattr(error, name, None)
        if value is not None:
            record[name] = value
    if isinstance(error, HTTPError):
        record["http_status"] = error.code
    cause = error.__cause__
    if isinstance(error, URLError) and isinstance(error.reason, BaseException):
        cause = error.reason
    if cause is not None and _depth < 8:
        record["cause"] = error_record(cause, stage=stage, _depth=_depth + 1)
    return record


def failure_guidance(failure_class: str | None) -> tuple[str, str]:
    """Return a concise cause and a concrete recovery action for the UI."""

    try:
        return FailureClass(str(failure_class or FailureClass.UNKNOWN_ERROR)).guidance
    except ValueError:
        return FailureClass.UNKNOWN_ERROR.guidance
