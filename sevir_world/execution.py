from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from .safety import UnsafeCodeError, validate_code


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().with_name("safe_worker.py")
EXECUTION_LOG_ROOT = ROOT / "data" / "executions"


def _persist_execution(
    code: str,
    context: dict[str, Any],
    result: dict[str, Any],
) -> str:
    """Persist one bounded execution record and return its opaque ID."""
    EXECUTION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex
    record = {
        "execution_id": execution_id,
        "created_at_epoch_seconds": round(time.time(), 3),
        "code": code,
        "context": context,
        "execution": result,
    }
    path = EXECUTION_LOG_ROOT / f"{execution_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return execution_id


class SafeExecutionClient:
    """Run validated SEVIRWorld code in a monitored isolated subprocess."""

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        memory_limit_mb: int = 512,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.memory_limit_mb = int(memory_limit_mb)

    def execute(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate, execute, and return structured result/error/provenance."""
        started = time.perf_counter()
        bound_context = dict(context or {})
        try:
            validate_code(code)
        except UnsafeCodeError as error:
            result = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_id"] = _persist_execution(code, bound_context, result)
            return result

        payload = json.dumps({"code": code, "context": bound_context}, ensure_ascii=False)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
        }
        environment["PYTHONIOENCODING"] = "utf-8"
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryDirectory(prefix="sevir-direct-") as workdir:
            environment["MPLCONFIGDIR"] = workdir
            process = subprocess.Popen(
                [sys.executable, "-I", str(WORKER)],
                cwd=workdir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
            memory_exceeded = threading.Event()
            stop_monitor = threading.Event()

            def monitor() -> None:
                try:
                    root_process = psutil.Process(process.pid)
                    while not stop_monitor.wait(0.05) and process.poll() is None:
                        processes = [root_process, *root_process.children(recursive=True)]
                        rss = sum(item.memory_info().rss for item in processes if item.is_running())
                        if rss > self.memory_limit_mb * 1024 * 1024:
                            memory_exceeded.set()
                            process.kill()
                            return
                except (psutil.Error, OSError):
                    return

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            try:
                stdout, stderr = process.communicate(payload, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                stop_monitor.set()
                result = {
                    "status": "error",
                    "error_type": "ExecutionTimeout",
                    "error": f"execution exceeded {self.timeout_seconds:g} seconds",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
                result["execution_id"] = _persist_execution(code, bound_context, result)
                return result
            finally:
                stop_monitor.set()
                monitor_thread.join(timeout=0.5)
        if memory_exceeded.is_set():
            result = {
                "status": "error",
                "error_type": "MemoryLimitExceeded",
                "error": f"execution exceeded {self.memory_limit_mb} MiB",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_id"] = _persist_execution(code, bound_context, result)
            return result
        if process.returncode != 0:
            result = {
                "status": "error",
                "error_type": "WorkerProcessError",
                "error": stderr.strip()[:4_000] or f"worker exited with code {process.returncode}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result["execution_id"] = _persist_execution(code, bound_context, result)
            return result
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as error:
            result = {
                "status": "error",
                "error_type": "WorkerProtocolError",
                "error": f"invalid worker JSON: {error}",
                "stderr": stderr.strip()[:4_000],
            }
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        result["limits"] = {
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_mb": self.memory_limit_mb,
            "network": "not exposed to generated code",
            "filesystem": "not exposed; approved artifact API only",
            "imports": "disabled",
        }
        result["execution_id"] = _persist_execution(code, bound_context, result)
        return result
