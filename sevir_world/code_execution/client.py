from __future__ import annotations

import json
import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from ..agent.contracts import (
    ExecutionRequest,
    ExecutionRuntime,
    WORKER_CONTRACT_VERSION,
)
from ..request.execution import build_execution_request, data_descs_from_payload
from .safety import UnsafeCodeError, normalize_generated_code, validate_code
from ..capabilities.registry import allowed_capabilities, execution_requirements, registry_revision
from ..execution_state import invalidated_state


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().with_name("worker.py")
_DEFAULT_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "sevir_world_runtime"
EXECUTION_LOG_ROOT = Path(os.getenv("SEVIR_EXECUTION_LOG_DIR", str(_DEFAULT_RUNTIME_ROOT / "executions")))
_DEFAULT_ARTIFACT_ROOT = _DEFAULT_RUNTIME_ROOT / "artifacts"
try:
    _DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SEVIR_ARTIFACT_DIR", str(_DEFAULT_ARTIFACT_ROOT))
except OSError:
    fallback_artifacts = _DEFAULT_RUNTIME_ROOT / "artifacts"
    fallback_artifacts.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SEVIR_ARTIFACT_DIR", str(fallback_artifacts))


class _BestEffortTemporaryDirectory:
    """Temporary worker directory whose cleanup cannot mask execution output."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self.path: str | None = None

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self._prefix)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.path:
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except OSError:
                pass
        return False


def _persist_execution(
    code: str,
    request: ExecutionRequest,
    result: dict[str, Any],
) -> str:
    """Persist one bounded execution record and return its opaque ID."""
    try:
        EXECUTION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only demo environments still return the execution result; the
        # opaque provenance id is simply unavailable until a writable log path
        # is configured with SEVIR_EXECUTION_LOG_DIR.
        return ""
    execution_id = uuid.uuid4().hex
    record = {
        "execution_id": execution_id,
        "created_at_epoch_seconds": round(time.time(), 3),
        "code": code,
        "data_descs": request.data_descs,
        "selected_tools": request.selected_tools,
        "active_tools": request.active_tools,
        "generate_visual": request.generate_visual,
        "execution": result,
    }
    path = EXECUTION_LOG_ROOT / f"{execution_id}.json"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        return ""
    return execution_id


def _register_artifact_paths(
    value: Any,
    *,
    created_after: float | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Turn generated PNG paths into stable HTTP artifacts.

    File naming is runtime infrastructure, not an analytical task. Generated
    code may save any PNG name below the configured artifact directory; the
    parent registers a content file under an opaque UUID and replaces the path
    in the observation before the next Agent step.
    """
    root_text = str(os.getenv("SEVIR_ARTIFACT_DIR", "")).strip()
    if not root_text:
        return value, []
    root = Path(root_text).resolve()
    artifacts: list[dict[str, Any]] = []
    registered: dict[str, str] = {}

    def renderable_png(path: Path) -> bool:
        """Reject zero-byte/truncated PNGs before exposing them as evidence."""
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                return False
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            return width > 1 and height > 1
        except (OSError, ValueError):
            return False

    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, list):
            return [convert(child) for child in item]
        if not isinstance(item, str) or not item.lower().endswith(".png"):
            return item
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate.name
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return item
        if not resolved.is_file() or not renderable_png(resolved):
            return item
        source_key = str(resolved)
        if source_key in registered:
            return registered[source_key]
        if re.fullmatch(r"[0-9a-f]{32}\.png", resolved.name):
            target = resolved
        else:
            target = root / f"{uuid.uuid4().hex}.png"
            shutil.copy2(resolved, target)
        url = f"/api/artifacts/{target.name}"
        registered[source_key] = url
        if any(item.get("url") == url for item in artifacts):
            return url
        artifacts.append({
            "artifact_id": target.stem,
            "url": url,
            "image_url": url,
            "title": resolved.stem,
            "description": "由本轮执行代码生成。",
            "chart_type": "agent_generated",
            "status": "completed",
        })
        return url

    normalized = convert(value)
    # Generated code sometimes saves a figure but forgets to include its path
    # in ``result``.  The worker contract is filesystem-backed, so register
    # PNGs present in the request artifact directory as well; this prevents a
    # real figure from being mistaken for missing evidence and retried.
    for candidate in sorted(root.glob("*.png")):
        if candidate.is_file() and renderable_png(candidate) and (
            created_after is None or candidate.stat().st_mtime >= created_after - 0.5
        ):
            convert(str(candidate))
    return normalized, artifacts


class ExecutionClient:
    """Run standard Python in a monitored isolated subprocess.

    Safety comes from process isolation, deadlines, memory monitoring and a
    temporary cwd, not from a partial Python dialect.
    """

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        memory_limit_mb: int = 512,
        live_timeout_seconds: float | None = None,
        live_memory_limit_mb: int | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = int(memory_limit_mb)
        self.live_timeout_seconds = float(
            live_timeout_seconds
            if live_timeout_seconds is not None
            else os.getenv("SEVIR_LIVE_EXECUTION_TIMEOUT_SECONDS", "1200")
        )
        self.live_memory_limit_mb = int(
            live_memory_limit_mb
            if live_memory_limit_mb is not None
            else os.getenv("SEVIR_LIVE_EXECUTION_MEMORY_MB", "12288")
        )
        # One persistent worker/namespace is the reuse mechanism during an
        # Agent run. Model/data products keep their native caches; answers and
        # transport envelopes are deliberately never cached here.
        self._worker_process: subprocess.Popen[str] | None = None
        self._worker_workdir: str | None = None
        self._worker_configuration_key: str | None = None
        self._worker_output_queue: queue.Queue[str] = queue.Queue()
        self._worker_stderr: list[str] = []
        self._worker_monitor_stop = threading.Event()
        self._worker_memory_exceeded = threading.Event()
        self._execute_lock = threading.Lock()

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        try:
            root_process = psutil.Process(process.pid)
            descendants = root_process.children(recursive=True)
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            root_process.kill()
        except psutil.Error:
            try:
                process.kill()
            except OSError:
                pass

    def close(self) -> None:
        process = self._worker_process
        self._worker_process = None
        self._worker_configuration_key = None
        self._worker_monitor_stop.set()
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                self._kill_process_tree(process)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self._worker_workdir:
            try:
                shutil.rmtree(self._worker_workdir, ignore_errors=True)
            except OSError:
                pass
            self._worker_workdir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _start_worker(
        self,
        environment: dict[str, str],
        configuration_key: str,
        memory_limit_mb: int,
    ) -> subprocess.Popen[str]:
        self.close()
        self._worker_monitor_stop = threading.Event()
        self._worker_memory_exceeded = threading.Event()
        self._worker_output_queue = queue.Queue()
        self._worker_stderr = []
        self._worker_workdir = tempfile.mkdtemp(prefix="sevir-session-")
        environment = dict(environment)
        environment["MPLCONFIGDIR"] = self._worker_workdir
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [sys.executable, "-I", str(WORKER), "--persistent"],
            cwd=self._worker_workdir,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._worker_process = process
        self._worker_configuration_key = configuration_key

        def pump_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self._worker_output_queue.put(line.rstrip("\r\n"))

        def pump_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                self._worker_stderr.append(line)
                if len(self._worker_stderr) > 200:
                    del self._worker_stderr[:50]

        def monitor() -> None:
            try:
                root_process = psutil.Process(process.pid)
                while not self._worker_monitor_stop.wait(0.05) and process.poll() is None:
                    processes = [root_process, *root_process.children(recursive=True)]
                    rss = sum(item.memory_info().rss for item in processes if item.is_running())
                    if rss > memory_limit_mb * 1024 * 1024:
                        self._worker_memory_exceeded.set()
                        self._kill_process_tree(process)
                        return
            except (psutil.Error, OSError):
                return

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()
        threading.Thread(target=monitor, daemon=True).start()
        return process

    def _persistent_worker_response(
        self,
        payload: str,
        environment: dict[str, str],
        configuration_key: str,
        timeout_seconds: float,
        memory_limit_mb: int,
        is_cancelled: Any,
    ) -> tuple[str | None, str | None]:
        process = self._worker_process
        if (
            process is None
            or process.poll() is not None
            or self._worker_configuration_key != configuration_key
        ):
            process = self._start_worker(environment, configuration_key, memory_limit_mb)
        try:
            assert process.stdin is not None
            # The pipe carries a control message. Evidence and generated code
            # stay complete in the worker-owned directory until execution ends.
            Path(self._worker_workdir, "request.json").write_text(payload, encoding="utf-8")
            process.stdin.write('{"command":"execute_file"}\n')
            process.stdin.flush()
        except OSError as error:
            self.close()
            return None, f"worker input failed: {error}"

        communication_deadline = time.monotonic() + timeout_seconds
        while True:
            if is_cancelled():
                self.close()
                return None, "cancelled"
            if self._worker_memory_exceeded.is_set():
                self.close()
                return None, "memory"
            if process.poll() is not None and self._worker_output_queue.empty():
                stderr = "".join(self._worker_stderr).strip()[:4_000]
                self.close()
                return None, stderr or f"worker exited with code {process.returncode}"
            remaining = communication_deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                return None, "timeout"
            try:
                return self._worker_output_queue.get(timeout=min(0.1, remaining)), None
            except queue.Empty:
                continue

    def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        *,
        data_descs: list[dict[str, Any]] | None = None,
        selected_tools: list[str] | tuple[str, ...] | None = None,
        execution_request: ExecutionRequest | None = None,
        runtime: ExecutionRuntime | None = None,
    ) -> dict[str, Any]:
        """Validate, execute, and return structured result/error/provenance."""
        started = time.perf_counter()
        artifact_started_at = time.time()
        request = execution_request or build_execution_request(
            data_descs if data_descs is not None else data_descs_from_payload(context),
            selected_tools,
            context,
            dict((context or {}).get("_study_case") or {}),
        )
        # Some OpenAI-compatible gateways occasionally return a lone UTF-16
        # surrogate in generated source.  It is not valid UTF-8 and would make
        # the isolated-worker pipe fail before validation.  Replacement keeps
        # the source/provenance inspectable and lets the normal syntax checker
        # produce an actionable repair if the character mattered.
        code = str(code or "").encode("utf-8", errors="replace").decode("utf-8")
        code = normalize_generated_code(code)
        bound_context = dict(context or {})
        active_runtime = runtime or ExecutionRuntime()
        is_cancelled = active_runtime.is_cancelled
        if is_cancelled():
            result = {
                "status": "error",
                "error_type": "ExecutionCancelled",
                "error": "request was cancelled before worker execution",
                "error_phase": "request_cancelled",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_id"] = _persist_execution(code, request, result)
            return result
        # The worker profile is a property of the request contract, never of
        # the current generated program.  Classifying each repair from its
        # source text restarted the persistent worker whenever a later step
        # reused ``env`` without repeating ``aurora.regional_fields()``.
        # All request-bound weather capabilities therefore share one stable
        # extended profile for the lifetime of this ExecutionClient.
        requirements = execution_requirements(request.selected_tools)
        live_execution = request.binding.policy.mode == "live" or requirements.profile == "extended"
        timeout_seconds = self.live_timeout_seconds if live_execution else self.timeout_seconds
        memory_limit_mb = self.live_memory_limit_mb if live_execution else self.memory_limit_mb
        remaining_request_seconds = active_runtime.remaining_seconds()
        if remaining_request_seconds is not None:
            if remaining_request_seconds <= 0:
                result = {
                    "status": "error",
                    "error_type": "ExecutionTimeout",
                    "error": "request deadline reached before worker execution",
                    "error_phase": "request_deadline",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
                result["execution_id"] = _persist_execution(code, request, result)
                return result
            # Live tools previously inherited a 1200-second worker timeout,
            # allowing one generated program to outlive the 30-second Agent
            # request.  Every worker now consumes the same request deadline.
            timeout_seconds = min(timeout_seconds, max(0.1, remaining_request_seconds))
        try:
            validate_code(code, request.selected_tools)
        except UnsafeCodeError as error:
            result = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "error_phase": "code_preflight",
                "preflight": {
                    "status": "rejected",
                    "contract": "isolated_standard_python",
                    "rule": str(error),
                    "execution_started": False,
                },
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_id"] = _persist_execution(code, request, result)
            return result

        payload = json.dumps(
            {
                "code": code,
                "data_descs": request.data_descs,
                "selected_tools": request.selected_tools,
                "active_tools": request.active_tools,
                "binding": asdict(request.binding),
                "evidence_state": request.evidence_state,
                "generate_visual": request.generate_visual,
            },
            # Keep the parent -> isolated-worker protocol ASCII-only too.  On
            # Windows, redirected text pipes can otherwise split or recode a
            # multibyte character in generated chart labels/titles.  The
            # worker then sees syntactically invalid JSON before it can run
            # any tool.  JSON escapes are decoded back to the original Unicode
            # source by ``json.loads`` in the worker, so this changes only the
            # transport, not the generated program.
            ensure_ascii=True,
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {
                # Basic process/runtime paths and external-service credentials. Without USERPROFILE (or
                # CDSAPI_RC) an isolated worker cannot see the user's CDS
                # credentials even though the parent service can.
                "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOME",
                "CDSAPI_RC", "CDSAPI_URL", "CDSAPI_KEY", "CUDA_VISIBLE_DEVICES",
                "SEVIR_FOUNDATION_LIVE_ONLY", "SEVIR_CACHE_ONLY", "SEVIR_USE_VENDOR",
                "SEVIR_ARTIFACT_DIR",
                "SEVIR_STATE_MAX_OBJECTS", "SEVIR_STATE_MAX_BYTES",
            }
        }
        # A provider can request named configuration; only deployment can authorize it.
        deployment_environment = {
            key.strip().upper() for key in os.getenv(
                "SEVIR_WORKER_ALLOWED_ENV", "CASCAST_PYTHON,AURORA_PYTHON"
            ).split(",") if key.strip()
        }
        for key in requirements.environment:
            if key.upper() in deployment_environment and key in os.environ:
                environment[key] = os.environ[key]
        environment["SEVIR_AGENT_ALLOWED_CAPABILITIES"] = ",".join(allowed_capabilities())
        data_policy = request.binding.policy.mode
        if data_policy == "cached":
            # A cached demo must never turn a cache miss into an implicit
            # download or expensive model run.
            environment["SEVIR_CACHE_ONLY"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["SEVIR_WORKER_SELECTED_TOOLS"] = ",".join(request.selected_tools)
        configuration_key = hashlib.sha256(
            json.dumps(
                {
                    "worker_contract_version": WORKER_CONTRACT_VERSION,
                    "registry_revision": registry_revision(),
                    "request": {
                        "data_descs": request.data_descs,
                        "selected_tools": request.selected_tools,
                        "binding": asdict(request.binding),
                        "evidence_state": request.evidence_state,
                        "generate_visual": request.generate_visual,
                    },
                    "environment": environment,
                    "memory_limit_mb": memory_limit_mb,
                },
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        with self._execute_lock:
            stdout, worker_failure = self._persistent_worker_response(
                payload,
                environment,
                configuration_key,
                timeout_seconds,
                memory_limit_mb,
                is_cancelled,
            )
        if worker_failure == "memory":
            result = {
                "status": "error",
                "error_type": "MemoryLimitExceeded",
                "error": f"execution exceeded {memory_limit_mb} MiB",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_state"] = invalidated_state(worker_failure)
            result["execution_id"] = _persist_execution(code, request, result)
            return result
        if worker_failure == "cancelled":
            result = {
                "status": "error",
                "error_type": "ExecutionCancelled",
                "error": "request was cancelled during worker execution",
                "error_phase": "request_cancelled",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_state"] = invalidated_state(worker_failure)
            result["execution_id"] = _persist_execution(code, request, result)
            return result
        if worker_failure == "timeout":
            result = {
                "status": "error",
                "error_type": "ExecutionTimeout",
                "error": f"execution exceeded {timeout_seconds:g} seconds",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_state"] = invalidated_state(worker_failure)
            result["execution_id"] = _persist_execution(code, request, result)
            return result
        if worker_failure is not None or stdout is None:
            result = {
                "status": "error",
                "error_type": "WorkerProcessError",
                "error": worker_failure or "worker returned no response",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_state"] = invalidated_state(worker_failure or "no_response")
            result["execution_id"] = _persist_execution(code, request, result)
            return result
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as error:
            preview = stdout[:500]
            if len(stdout) > 500:
                preview += "...[truncated]"
            result = {
                "status": "error",
                "error_type": "WorkerProtocolError",
                "error": f"invalid worker JSON: {error}",
                "error_phase": "response_decode",
                "worker_stdout_characters": len(stdout),
                "worker_stdout_preview": preview,
                "stderr": "".join(self._worker_stderr).strip()[:4_000],
            }
            self.close()
            result["execution_state"] = invalidated_state("response_decode")
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        # A plotting program can save a complete figure and then fail in a
        # later scalar calculation.  Artifact discovery is independent of
        # Python's terminal status, so retain such evidence for the repair
        # turn instead of silently discarding it.
        normalized_result, generated_artifacts = _register_artifact_paths(
            result.get("result"), created_after=artifact_started_at
        )
        if "result" in result:
            result["result"] = normalized_result
        if generated_artifacts and not request.generate_visual:
            result.update({
                "status": "error",
                "error_type": "VisualArtifactForbidden",
                "error": (
                    "generate_visual is false; image artifacts are forbidden for this request"
                ),
                "error_phase": "artifact_validation",
            })
            generated_artifacts = []
        if generated_artifacts:
            result["generated_artifacts"] = generated_artifacts
        result["limits"] = {
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
            "budget_profile": "extended" if live_execution else "standard",
            "network": "inherits worker process policy; weather tools own supported data access",
            "filesystem": "temporary cwd; persistent figures use SEVIR_ARTIFACT_DIR",
            "imports": "standard Python imports enabled",
        }
        result["execution_id"] = _persist_execution(code, request, result)
        return result
