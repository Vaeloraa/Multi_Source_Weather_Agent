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
from dataclasses import fields, is_dataclass
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

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import scipy  # noqa: E402
import skimage  # noqa: E402

from sevir_world.code_execution.safety import (  # noqa: E402
    UnsafeCodeError,
    normalize_generated_code,
    validate_code,
    validate_imports,
)
from sevir_world.agent.contracts import (  # noqa: E402
    ExecutionRequest,
    WORKER_CONTRACT_VERSION,
)
from sevir_world.agent.runtime import ExecutionEnvironment  # noqa: E402
from sevir_world.agent.diagnostics import classify_execution_error  # noqa: E402


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

_HIDDEN_NAMESPACE_NAMES = {
    "np", "pd", "xr", "scipy", "skimage", "math", "ds", "study",
    "sevir", "geolocator", "cascast", "aurora", "plot_field", "result", "output",
    "__builtins__",
}


def _available_namespace_names(namespace: dict[str, Any] | None) -> list[str]:
    """Return concise user-visible names retained by the persistent Worker."""

    if not namespace:
        return []
    return sorted(
        name
        for name in namespace
        if not name.startswith("__") and name not in _HIDDEN_NAMESPACE_NAMES
    )[-100:]


def _namespace_value_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if type(value).__module__.startswith("xarray"):
        summary["dimensions"] = [str(item) for item in getattr(value, "dims", ())]
        summary["shape"] = [int(item) for item in getattr(value, "shape", ())]
    elif isinstance(value, np.ndarray):
        summary["shape"] = [int(item) for item in value.shape]
        summary["dtype"] = str(value.dtype)
    elif isinstance(value, pd.DataFrame):
        summary["rows"] = int(len(value))
        summary["columns"] = [str(item) for item in value.columns[:30]]
    elif isinstance(value, (list, tuple, dict, set)):
        summary["count"] = int(len(value))
    return summary


def _top_level_assignment_lines(tree: Any) -> dict[str, int]:
    """Return conservative module-level assignment names and first source lines."""

    if tree is None:
        return {}
    found: dict[str, int] = {}

    def targets(node: Any) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [name for item in node.elts for name in targets(item)]
        return []

    def visit_statements(statements: list[Any]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.setdefault(statement.name, int(statement.lineno))
                continue
            assignment_targets: list[Any] = []
            if isinstance(statement, ast.Assign):
                assignment_targets = list(statement.targets)
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                assignment_targets = [statement.target]
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                assignment_targets = [statement.target]
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                assignment_targets = [
                    item.optional_vars for item in statement.items
                    if item.optional_vars is not None
                ]
            for target in assignment_targets:
                for name in targets(target):
                    found.setdefault(name, int(statement.lineno))
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list):
                    visit_statements(nested)
            for handler in getattr(statement, "handlers", ()):
                if getattr(handler, "name", None):
                    found.setdefault(str(handler.name), int(handler.lineno))
                visit_statements(list(getattr(handler, "body", ())))

    visit_statements(list(getattr(tree, "body", ())))
    return found


def _namespace_state(
    before: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    tree: Any,
    *,
    committed: bool,
    error_line: int | None = None,
) -> dict[str, Any]:
    before = before or {}
    candidate = candidate or before
    safe_namespace = candidate if committed else before
    safe_names = _available_namespace_names(safe_namespace)
    missing = object()
    changed = sorted(
        name for name in candidate
        if not name.startswith("__")
        and name not in _HIDDEN_NAMESPACE_NAMES
        and before.get(name, missing) is not candidate.get(name, missing)
    )
    assignments = _top_level_assignment_lines(tree)
    created_or_updated = sorted({
        *changed,
        *(
            name for name in assignments
            if name in safe_namespace and name not in _HIDDEN_NAMESPACE_NAMES
        ),
    }) if committed else []
    retained = sorted(
        name for name in safe_names
        if name in before and name not in created_or_updated
    )
    not_reached = sorted(
        name for name, line in assignments.items()
        if error_line is not None and line > error_line
    )
    return {
        "transaction": "committed" if committed else "rolled_back",
        "safe_to_reuse": safe_names,
        "available_values": {
            name: _namespace_value_summary(safe_namespace[name])
            for name in safe_names[:40]
        },
        "retained_names": retained,
        "created_or_updated_names": created_or_updated,
        "attempted_assignment_names": sorted(assignments),
        "uncommitted_names": sorted(assignments) if not committed else [],
        "materialized_uncommitted_names": changed if not committed else [],
        "not_reached_assignment_names": not_reached if not committed else [],
        "result_slot": (
            "committed"
            if committed and ("result" in candidate or "output" in candidate)
            else "rolled_back"
            if not committed and ("result" in candidate or "output" in candidate)
            else "not_assigned"
        ),
    }


def _failed_value_shapes(
    working_namespace: dict[str, Any] | None,
    committed_namespace: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe generated values that existed only in a failed transaction.

    The values themselves are deliberately not committed. Their dimensions
    and shapes are nevertheless essential evidence for repairing generic
    xarray/numpy/matplotlib mistakes without guessing or rerunning a tool.
    """
    if not working_namespace:
        return {}
    committed = committed_namespace or {}
    descriptions: dict[str, Any] = {}
    for name, value in working_namespace.items():
        if name.startswith("__") or name in _HIDDEN_NAMESPACE_NAMES or committed.get(name) is value:
            continue
        if type(value).__module__.startswith("xarray"):
            descriptions[name] = {
                "type": type(value).__name__,
                "dimensions": [str(item) for item in getattr(value, "dims", ())],
                "shape": [int(item) for item in getattr(value, "shape", ())],
                "committed": False,
            }
        elif isinstance(value, np.ndarray):
            descriptions[name] = {
                "type": "ndarray",
                "shape": [int(item) for item in value.shape],
                "committed": False,
            }
    return descriptions


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
    if type(value).__module__.startswith("xarray"):
        data_vars = getattr(value, "data_vars", None)
        if data_vars is not None:
            return {
                "type": type(value).__name__,
                "representation": "metadata_only",
                "read_scope": "worker_only",
                "dimensions": {str(key): int(size) for key, size in value.sizes.items()},
                "variables": [str(name) for name in data_vars],
                "coordinates": [str(name) for name in value.coords],
                "values_omitted": True,
            }
        return {
            "type": type(value).__name__,
            "representation": "metadata_only",
            "read_scope": "worker_only",
            "name": str(getattr(value, "name", "") or ""),
            "dimensions": [str(name) for name in getattr(value, "dims", ())],
            "shape": [int(size) for size in getattr(value, "shape", ())],
            "dtype": str(getattr(value, "dtype", "")),
            "values_omitted": True,
        }
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "representation": "metadata_only",
            "read_scope": "worker_only",
            "shape": [int(item) for item in value.shape],
            "columns": [str(item) for item in value.columns],
            "rows_omitted": True,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name), _depth + 1)
            for field in fields(value)
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


def _capture_open_figures() -> None:
    """Persist figures created by Agent code even when it omitted savefig()."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    root_text = str(os.getenv("SEVIR_ARTIFACT_DIR", "")).strip()
    if pyplot is None or not root_text:
        return
    root = Path(root_text)
    root.mkdir(parents=True, exist_ok=True)
    for index, number in enumerate(list(pyplot.get_fignums()), start=1):
        figure = pyplot.figure(number)
        figure.savefig(root / f"agent_figure_{index}.png", dpi=120, bbox_inches="tight")


class WorkerSession:
    """One request-scoped Python namespace shared by every execute step."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.request: ExecutionRequest | None = None
        self.environment: ExecutionEnvironment | None = None
        self.runtime_schema: dict[str, Any] | None = None
        self.namespace: dict[str, Any] | None = None
        self.step = 0

    def bind(self, request: ExecutionRequest) -> None:
        if self.request is not None:
            if request != self.request:
                raise ValueError(
                    "persistent worker cannot change request binding or selected resources "
                    "inside one execution session"
                )
            return
        self.request = request
        self.environment = ExecutionEnvironment(request)
        self.runtime_schema = self.environment.describe()
        self.runtime_schema["worker_contract_version"] = WORKER_CONTRACT_VERSION
        self.namespace = {
            "__builtins__": builtins.__dict__,
            "np": np,
            "pd": pd,
            "xr": xr,
            "scipy": scipy,
            "skimage": skimage,
            "math": math,
            **self.environment.resources,
        }


def _execute_payload(session: WorkerSession, payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any]
    environment = session.environment
    runtime_schema = session.runtime_schema
    phase = "request_read"
    code = ""
    captured = BoundedOutput()
    tree: Any = None
    before_namespace: dict[str, Any] | None = None
    working_namespace: dict[str, Any] | None = None
    call_start = len(environment.recorder.calls) if environment else 0
    try:
        phase = "request_validation"
        code = str(payload.get("code", ""))
        code = normalize_generated_code(code)
        request = ExecutionRequest.from_payload(
            list(payload.get("data_descs") or []),
            list(payload.get("selected_tools") or []),
            dict(payload.get("binding") or {}),
            dict(payload.get("evidence_state") or {}),
            visual_artifact_required=bool(
                payload.get("visual_artifact_required", False)
            ),
        )
        tree = validate_code(code, request.selected_tools)
        phase = "environment_setup"
        session.bind(request)
        environment = session.environment
        runtime_schema = session.runtime_schema
        validate_imports(tree, tuple(runtime_schema.get("available_modules", ())))
        namespace = session.namespace
        assert environment is not None and namespace is not None
        before_namespace = dict(namespace)
        call_start = len(environment.recorder.calls)
        def safe_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
            captured.write(sep.join(str(value) for value in values) + end)

        # Preserve working variables across reflective steps, but require each
        # generated program to publish its own output. Otherwise a step that
        # forgot to assign ``result`` silently returned the previous step's
        # value and made persistence look like a successful new execution.
        working_namespace = dict(namespace)
        working_namespace.pop("result", None)
        working_namespace.pop("output", None)
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(tree, "<sevir-agent>", "exec"), working_namespace)
            phase = "generated_exec"
            _capture_open_figures()
            result_value = working_namespace.get("result", working_namespace.get("output"))
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
        # Commit variables only after execution and result serialization both
        # succeed. A failed reflective attempt may acquire useful tool evidence,
        # but its half-created Python variables must never poison the next step.
        namespace.clear()
        namespace.update(working_namespace)
        calls = _json_safe(environment.recorder.calls[call_start:])
        successful_calls = sum(
            isinstance(item, dict) and item.get("status") == "ok" for item in calls
        )
        failed_calls = sum(
            not isinstance(item, dict) or item.get("status") != "ok" for item in calls
        )
        observation_quality = "partial" if failed_calls else "complete"
        output = {
            "status": "partial" if failed_calls else "ok",
            "observation_quality": observation_quality,
            "result": result,
            "primary_result": _json_safe(_primary_result(calls, result)),
            "calls": calls,
            "tool_call_summary": {
                "total": len(calls),
                "successful": successful_calls,
                "failed": failed_calls,
            },
            "tool_cost_summary": environment.recorder.summary(),
            "evidence_acquired": successful_calls > 0,
            "evidence_tool_call_count": successful_calls,
            "postprocessing_succeeded": True,
            "stdout": captured.getvalue(),
            "result_json_characters": result_characters,
            "runtime_schema": runtime_schema,
            "available_names": _available_namespace_names(namespace),
            "namespace_state": _namespace_state(
                before_namespace, namespace, tree, committed=True
            ),
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
        failed_step_calls = (
            _json_safe(environment.recorder.calls[call_start:]) if environment else []
        )
        successful_calls = sum(
            isinstance(item, dict) and item.get("status") == "ok"
            for item in failed_step_calls
        )
        failed_calls = len(failed_step_calls) - successful_calls
        output = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "error_phase": phase,
            # Preserve the complete traceback for the repair loop. It stays in
            # the technical trace and is never copied verbatim to the user.
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-12000:],
            "error_location": error_location,
            # A failed program may have completed useful calculations and
            # printed their compact values before the exception.  Keeping
            # that evidence lets the next reflective turn repair one line
            # instead of repeating every successful tool call.
            "stdout": captured.getvalue(),
            "calls": failed_step_calls,
            "tool_call_summary": {
                "total": len(failed_step_calls),
                "successful": successful_calls,
                "failed": failed_calls,
            },
            "tool_cost_summary": environment.recorder.summary() if environment else None,
            "evidence_acquired": successful_calls > 0,
            "evidence_tool_call_count": successful_calls,
            "postprocessing_succeeded": False,
            "available_names": _available_namespace_names(
                session.namespace if session is not None else None
            ),
            "namespace_state": _namespace_state(
                before_namespace,
                working_namespace,
                tree,
                committed=False,
                error_line=(error_location or {}).get("line"),
            ),
            "failed_value_shapes": _failed_value_shapes(
                working_namespace,
                session.namespace if session is not None else None,
            ),
            "runtime_schema": runtime_schema if environment else None,
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
