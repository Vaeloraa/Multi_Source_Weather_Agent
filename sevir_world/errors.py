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

    PROVIDER_QUOTA_EXHAUSTED = ('provider_quota_exhausted', '模型服务的可用额度已经耗尽。', '补充额度或切换到有额度的模型渠道后重试；已完成的上游数据不会被当作失败结论。')
    PROVIDER_MODEL_UNAVAILABLE = ('provider_model_unavailable', '当前渠道没有提供所配置的模型。', '从 /v1/models 返回的可用模型中选择一个，并核对项目环境变量中的模型名。')
    MODEL_PARAMETER_ERROR = ('model_parameter_error', '当前模型不接受请求中携带的某个参数。', '移除该模型不支持的参数后重试，例如仅在明确支持时发送 thinking 配置。')
    PROVIDER_AUTHENTICATION = ('provider_authentication', '模型服务拒绝了当前访问凭据。', '检查 API Key、请求地址和账户所属渠道；不要在前端或日志中再次暴露密钥。')
    TRANSPORT_TLS_ERROR = ('transport_tls_error', 'HTTPS/TLS 连接在传输过程中异常结束。', '核对代理、证书和网关稳定性后重试。')
    LOCAL_ASSET_MISSING = ('local_asset_missing', '执行所需的本地文件或能力模块不存在。', '检查错误中给出的完整路径，并恢复缺失文件或修正项目路径配置。')
    CACHE_MISS = ('cache_miss', '没有找到与当前事件、时间和输入完全匹配的缓存。', '若策略允许现场运行，应继续实时获取或推理；完成后把结果写入相同缓存键。')
    TRANSPORT_CIRCUIT_OPEN = ('transport_circuit_open', '连续连接失败触发了暂时熔断。', '等待熔断窗口结束，或修复上游服务后再试。')
    PROVIDER_RATE_LIMIT = ('provider_rate_limit', '模型渠道当前过载或触发了频率限制。', '稍后重试或换用当前负载较低的可用渠道。')
    PROVIDER_5XX = ('provider_5xx', '模型服务端暂时异常。', '保留本轮已完成步骤，稍后从失败步骤重试。')
    TRANSPORT_CONNECT_TIMEOUT = ('transport_connect_timeout', '连接模型或数据服务时超时。', '检查网络和服务地址，然后从当前步骤重试。')
    EXECUTION_TIMEOUT = ('execution_timeout', '本地分析或推理超过了当前执行上限。', '检查卡住的工具步骤，必要时缩小范围或使用已经生成的缓存结果。')
    TRANSPORT_READ_TIMEOUT = ('transport_read_timeout', '远端服务已连接，但在规定时间内没有返回结果。', '确认服务仍在运行；现场推理可继续等待或主动停止后重试。')
    TRANSPORT_RESET = ('transport_reset', '远端连接在返回完成前被中断。', '检查模型网关状态并重试；已完成的工具证据会继续保留。')
    ANALYSIS_BUDGET_EXHAUSTED = ('analysis_budget_exhausted', 'Agent 已用完本轮允许的分析次数。', '基于现有证据先形成部分结论，或增加轮次后从最后一个未完成步骤继续。')
    CONTEXT_BUDGET_EXCEEDED = ('context_budget_exceeded', '当前问题和必要工具说明超过了模型输入预算。', '缩小本轮问题或工具范围，或按模型实际容量调整上下文预算。')
    RENDER_ERROR = ('render_error', '数据分析可能已经完成，但图表生成或校验失败。', '保留数值结果，并检查失败图表的输入范围、空数组和输出路径。')
    TOOL_ARGUMENT_ERROR = ('tool_argument_error', '工具收到的参数不符合能力契约。', '查看失败步骤中的工具名和参数，修正后只重跑该步骤。')
    TOOL_RUNTIME_ERROR = ('tool_runtime_error', '工具在真实执行过程中报错。', '展开失败步骤查看工具错误；上游已经成功的证据可以继续复用。')
    CLIENT_DISCONNECTED = ('client_disconnected', '浏览器连接已断开，本轮执行被取消。', '保持页面连接后重新提交；若结果已缓存，可直接复用缓存。')
    MODEL_PROTOCOL_ERROR = ('model_protocol_error', '模型没有返回 Agent 可执行的完整动作格式。', '重试该模型步骤，或换用能稳定遵循动作协议的模型。')
    # Legacy task records only; the current Agent does not run patch repair.
    CODE_REPAIR_FAILED = ('code_repair_failed', '历史任务的补丁修复未完成，已完成的步骤已保留。', '当前已移除补丁机制；查看原始错误，通过普通代码执行继续分析。')
    MODEL_TRANSPORT_FAILURE = ('model_transport_failure', '语言模型调用在返回完整结果前失败。', '展开模型步骤查看上游错误；已完成的工具证据可以保留并在重试时复用。')
    SOLUTION_EVIDENCE_ERROR = ('solution_evidence_error', '最终回答没有满足问题要求的证据链。', '查看缺失的观测、预测或历史依赖，只补跑缺失步骤后重新整理回答。')
    MISSING_REQUIRED_VISUAL = ('missing_required_visual', '本轮要求生成图片，但尚未生成有效的新图；已有文字和来源已保留。', '仅补做本轮要求的图表或示意图并复用已有资料；对话中的旧图不计入本轮。若只需文字，可关闭“生成图片”后重新提交。')
    CONSECUTIVE_EXECUTION_ERRORS = ('consecutive_execution_errors', '代码或工具连续执行失败，Agent 为避免无效重试而停止。', '展开最后几个执行步骤，先修复第一个真实工具错误再继续。')
    SERVICE_RESTARTED = ('service_restarted', '服务在任务处于非终态时发生了重启，原执行进程无法继续。', '已完成的缓存、会话和步骤快照仍保留；重新提交后会优先复用精确匹配的缓存。')
    ANALYSIS_INCOMPLETE = ('analysis_incomplete', '所需分析尚未全部完成。', '可根据已取得的结果缩小范围，或补充缺失的数据。')
    UNKNOWN_ERROR = ('unknown_error', '系统捕获到了尚未归类的异常。', '展开失败步骤查看原始错误和发生阶段，再决定是否重试。')


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
