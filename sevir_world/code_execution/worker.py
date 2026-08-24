from __future__ import annotations

import ast
import io
import json
import os
import sys
import traceback
import builtins
import contextlib
import uuid
from pathlib import Path
from typing import Any

import math
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
# The HTTP process may load binary dependencies from the repository's bundled
# vendor directory.  ``python -I`` intentionally ignores PYTHONPATH, so the
# isolated worker must bind the same trusted directory explicitly before it
# imports the application package.  Without this, h5py/pyproj were available
# to preflight but missing from every real generated-code execution.
VENDOR = ROOT / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from sevir_world.code_execution.safety import (  # noqa: E402
    UnsafeCodeError,
    normalize_generated_code,
    validate_code,
)
from sevir_world.agent.contracts import ExecutionRequest  # noqa: E402
from sevir_world.agent.runtime import ExecutionEnvironment  # noqa: E402
from sevir_world.agent.schema import classify_execution_error  # noqa: E402


class BoundedOutput(io.StringIO):
    def __init__(self, limit: int = 16_000) -> None:
        super().__init__()
        self.limit = limit

    def write(self, value: str) -> int:
        remaining = self.limit - self.tell()
        if remaining <= 0:
            return len(value)
        super().write(value[:remaining])
        return len(value)


MAX_INLINE_ARRAY_ELEMENTS = 4_096


def _json_safe(value: Any, _depth: int = 0) -> Any:
    if _depth > 40:
        raise ValueError("result nesting exceeds 40 levels")
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), _depth + 1)
    if isinstance(value, np.ndarray):
        if value.size <= MAX_INLINE_ARRAY_ELEMENTS:
            return _json_safe(value.tolist(), _depth + 1)
        return {
            "array_values_omitted": True,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "elements": int(value.size),
            "guidance": "return selected slices or arrays reductions instead of a raw grid",
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, _depth + 1) for item in value]
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist(), _depth + 1)
    raise TypeError(f"result contains unsupported type: {type(value).__name__}")


def _primary_result(calls: list[dict[str, Any]], result: Any) -> Any:
    if isinstance(result, dict):
        return result
    successful = [item.get("result") for item in calls if item.get("status") == "ok"]
    dictionaries = [item for item in successful if isinstance(item, dict)]
    if dictionaries:
        return dictionaries[-1]
    return result


class WorkerSession:
    """One request-scoped Python namespace shared by every execute step."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.request: ExecutionRequest | None = None
        self.environment: ExecutionEnvironment | None = None
        self.runtime_schema: dict[str, Any] | None = None
        self.namespace: dict[str, Any] | None = None
        self.step = 0

    def bind(self, request: ExecutionRequest, replay_results: dict[str, Any]) -> None:
        if self.request is not None:
            if request != self.request:
                raise ValueError(
                    "persistent worker cannot change data_descs or selected_tools "
                    "inside one execution session"
                )
            return
        self.request = request
        self.environment = ExecutionEnvironment(request, replay_results=replay_results)
        self.runtime_schema = self.environment.describe()
        self.namespace = {
            "__builtins__": builtins.__dict__,
            "np": np,
            "math": math,
            **self.environment.resources,
        }


def _execute_payload(session: WorkerSession, payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any]
    environment = session.environment
    runtime_schema = session.runtime_schema
    phase = "request_read"
    code = ""
    call_start = len(environment.recorder.calls) if environment else 0
    try:
        phase = "request_validation"
        code = str(payload.get("code", ""))
        code = normalize_generated_code(code)
        request = ExecutionRequest.from_payload(
            list(payload.get("data_descs") or []),
            list(payload.get("selected_tools") or []),
        )
        tree = validate_code(code, request.selected_tools)
        phase = "environment_setup"
        session.bind(request, dict(payload.get("tool_result_cache") or {}))
        environment = session.environment
        runtime_schema = session.runtime_schema
        namespace = session.namespace
        assert environment is not None and namespace is not None
        call_start = len(environment.recorder.calls)
        captured = BoundedOutput()

        def safe_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
            captured.write(sep.join(str(value) for value in values) + end)

        # A previous_result field is accepted only for compatibility with an
        # older parent. In a persistent session the real Python object already
        # exists in ``namespace`` and must never be replaced by a JSON copy.
        if session.step == 0 and payload.get("previous_result") is not None:
            namespace["result"] = payload["previous_result"]
        defines_run = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
            for node in tree.body
        )
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(tree, "<sevir-agent>", "exec"), namespace)
            run_function = namespace.get("run") if defines_run else None
            if defines_run and callable(run_function):
                phase = "generated_run"
                result_value = environment.invoke(run_function)
            else:
                phase = "generated_exec"
                result_value = namespace.get("result", namespace.get("output"))
            result = _json_safe(result_value)
        session.step += 1
        phase = "result_serialization"
        result_characters = len(json.dumps(result, ensure_ascii=True))
        max_result_characters = int(os.getenv("SEVIR_WORKER_MAX_RESULT_CHARACTERS", "1000000"))
        if result_characters > max_result_characters:
            raise ValueError(
                f"generated result is {result_characters:,} characters; "
                f"return compact calculations/artifacts under {max_result_characters:,} characters "
                "and keep raw grids inside the Worker"
            )
        calls = _json_safe(environment.recorder.calls[call_start:])
        output = {
            "status": "ok",
            "result": result,
            "primary_result": _json_safe(_primary_result(calls, result)),
            "calls": calls,
            "stdout": captured.getvalue(),
            "result_json_characters": result_characters,
            "runtime_schema": runtime_schema,
            "evidence_ledger": _json_safe(environment.recorder.ledger.prompt_context()),
            "tool_result_cache_updates": _json_safe(environment.recorder.reusable_results()),
            "worker_session": {"id": session.session_id, "step": session.step},
        }
    except Exception as error:
        generated_frame = next(
            (
                frame for frame in reversed(traceback.extract_tb(error.__traceback__))
                if frame.filename == "<sevir-agent>"
            ),
            None,
        )
        source_lines = code.splitlines()
        error_location = None
        if generated_frame is not None:
            line_number = int(generated_frame.lineno)
            source_line = (
                source_lines[line_number - 1].strip()
                if 0 < line_number <= len(source_lines)
                else str(generated_frame.line or "").strip()
            )
            error_location = {
                "line": line_number,
                "source": source_line,
                "function": str(generated_frame.name),
            }
        output = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "error_phase": phase,
            # Preserve the complete traceback for the repair loop. It stays in
            # the technical trace and is never copied verbatim to the user.
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-12000:],
            "error_location": error_location,
            "calls": _json_safe(environment.recorder.calls[call_start:]) if environment else [],
            "runtime_schema": runtime_schema if environment else None,
            "evidence_ledger": _json_safe(environment.recorder.ledger.prompt_context()) if environment else None,
            "tool_result_cache_updates": _json_safe(environment.recorder.reusable_results()) if environment else {},
            "repair": classify_execution_error(type(error).__name__, str(error)),
            "worker_session": {"id": session.session_id, "step": session.step},
        }
    return output


def _encode_output(output: dict[str, Any]) -> str:
    # ASCII transport avoids Windows redirected-console code-page ambiguity.
    encoded = json.dumps(output, ensure_ascii=True)
    max_output_characters = int(os.getenv("SEVIR_WORKER_MAX_OUTPUT_CHARACTERS", "4000000"))
    if len(encoded) > max_output_characters:
        encoded = json.dumps(
            {
                "status": "error",
                "error_type": "OutputLimitError",
                "error": f"execution result exceeds {max_output_characters:,} characters",
            },
            ensure_ascii=True,
        )
    return encoded


def main() -> None:
    persistent = "--persistent" in sys.argv[1:]
    session = WorkerSession()
    if not persistent:
        raw = sys.stdin.read(65_537)
        if len(raw) > 65_536:
            output = {
                "status": "error",
                "error_type": "ValueError",
                "error": "execution request exceeds 64 KiB",
                "error_phase": "request_read",
            }
        else:
            try:
                output = _execute_payload(session, json.loads(raw))
            except Exception as error:
                output = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "error_phase": "request_decode",
                }
        sys.stdout.write(_encode_output(output))
        return

    for raw in sys.stdin:
        raw = raw.rstrip("\r\n")
        if not raw:
            continue
        try:
            if len(raw) > 65_536:
                raise ValueError("execution request exceeds 64 KiB")
            payload = json.loads(raw)
            if payload.get("command") == "shutdown":
                break
            output = _execute_payload(session, payload)
        except Exception as error:
            output = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "error_phase": "request_decode",
                "worker_session": {"id": session.session_id, "step": session.step},
            }
        sys.stdout.write(_encode_output(output) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
