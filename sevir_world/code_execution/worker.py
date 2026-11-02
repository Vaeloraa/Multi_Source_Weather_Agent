from __future__ import annotations

import ast
import io
import json
import os
import sys
import traceback
import contextlib
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

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

from sevir_world.code_execution.safety import (  # noqa: E402
    UnsafeCodeError,
    normalize_generated_code,
    restricted_builtins,
    validate_code,
    validate_imports,
)
from sevir_world.agent.contracts import ExecutionRequest  # noqa: E402
from sevir_world.capabilities.base import RequestBinding  # noqa: E402
from sevir_world.capabilities.registry import validate_selected_capabilities  # noqa: E402
from sevir_world.serialization import serialize_value, value_metadata  # noqa: E402
from sevir_world.errors import error_record  # noqa: E402
from sevir_world.agent.runtime import ExecutionEnvironment  # noqa: E402
from sevir_world.execution_state import ObjectStore, compact_state, fork_namespace, state_receipt  # noqa: E402


MAX_INLINE_ARRAY_ELEMENTS = 4_096


def json_safe(value: Any, depth: int = 0) -> Any:
    """Convert execution output to JSON while keeping large scientific data local."""
    return serialize_value(value, inline_array_elements=MAX_INLINE_ARRAY_ELEMENTS, depth=depth)


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


_HIDDEN_NAMESPACE_NAMES = {
    "np", "pd", "xr", "scipy", "skimage", "math", "ds", "study", "case_memory", "analytics",
    "sevir", "geospatial", "cascast", "era5", "aurora", "plot_field", "result", "output",
    "__builtins__", "objects",
}


def _available_namespace_names(
    namespace: dict[str, Any] | None,
    injected_names: set[str] | None = None,
) -> list[str]:
    """Return concise user-visible names retained by the persistent Worker."""

    if not namespace:
        return []
    return sorted(
        name
        for name in namespace
        if not name.startswith("__")
        and name not in _HIDDEN_NAMESPACE_NAMES
        and name not in (injected_names or set())
    )[-100:]


def _namespace_value_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    metadata = value_metadata(value)
    if metadata is not None:
        summary.update(metadata)
        if isinstance(value, pd.DataFrame):
            summary["rows"] = len(value)
            summary["columns"] = summary["columns"][:30]
        if "variables" in metadata:
            summary["data_vars"] = metadata["variables"]
    elif is_dataclass(value) and not isinstance(value, type):
        summary["fields"] = [field.name for field in fields(value)][:40]
    elif isinstance(value, dict):
        summary["count"] = int(len(value))
        summary["keys"] = [str(item) for item in list(value)[:40]]
        summary["value_types"] = {
            str(key): type(item).__name__
            for key, item in list(value.items())[:20]
        }
    elif isinstance(value, (list, tuple, set)):
        summary["count"] = int(len(value))
        summary["element_types"] = sorted({
            type(item).__name__ for item in list(value)[:20]
        })
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
    injected_names: set[str] | None = None,
) -> dict[str, Any]:
    before = before or {}
    candidate = candidate or before
    safe_namespace = candidate if committed else before
    hidden = _HIDDEN_NAMESPACE_NAMES | (injected_names or set())
    safe_names = _available_namespace_names(safe_namespace, injected_names)
    missing = object()
    changed = sorted(
        name for name in candidate
        if not name.startswith("__")
        and name not in hidden
        and before.get(name, missing) is not candidate.get(name, missing)
    )
    assignments = _top_level_assignment_lines(tree)
    created_or_updated = sorted({
        *changed,
        *(
            name for name in assignments
            if name in safe_namespace and name not in hidden
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
    injected_names: set[str] | None = None,
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
        if (
            name.startswith("__")
            or name in _HIDDEN_NAMESPACE_NAMES
            or name in (injected_names or set())
            or committed.get(name) is value
        ):
            continue
        if (
            type(value).__module__.startswith("xarray")
            or isinstance(value, (np.ndarray, pd.DataFrame, pd.Series, dict, list, tuple))
            or (is_dataclass(value) and not isinstance(value, type))
        ):
            descriptions[name] = {
                **_namespace_value_summary(value),
                "committed": False,
            }
    return descriptions


def _primary_result(calls: list[dict[str, Any]], result: Any) -> Any:
    if isinstance(result, dict):
        return result
    successful = [item.get("result") for item in calls if item.get("status") == "ok"]
    dictionaries = [item for item in successful if isinstance(item, dict)]
    if dictionaries:
        return dictionaries[-1]
    return result


def _capture_open_figures(*, generate_visual: bool) -> None:
    """Persist figures created by Agent code even when it omitted savefig()."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    root_text = str(os.getenv("SEVIR_ARTIFACT_DIR", "")).strip()
    if pyplot is None or not root_text:
        return
    from sevir_world.tools.visualization.renderer import figure_path
    for index, number in enumerate(list(pyplot.get_fignums()), start=1):
        figure = pyplot.figure(number)
        if getattr(figure, "_sevir_saved", False):
            pyplot.close(figure)
            continue
        # Tick labels and spines are always present, even for a completely
        # empty ``plt.subplots()``.  They are not evidence.  Only capture a
        # figure that contains an actual plotted primitive; explicitly saved
        # figures are still handled by the client registration layer.
        has_content = any(
            axis.images or axis.lines or axis.collections or axis.patches
            for axis in figure.axes
        )
        if not has_content:
            pyplot.close(figure)
            continue
        if not generate_visual:
            pyplot.close("all")
            raise PermissionError(
                "generate_visual is false; image artifact generation is disabled"
            )
        figure.savefig(figure_path(f"agent_figure_{index}.png"), dpi=120, bbox_inches="tight")
        pyplot.close(figure)


def _configure_matplotlib(*, generate_visual: bool) -> None:
    """Configure plotting once per Worker binding and enforce managed artifact writes."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "DengXian", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        # Generated programs occasionally use Linux paths such as /tmp/x.png
        # or the read-only worker cwd.  Artifact placement is infrastructure,
        # so redirect every filename-based savefig call into the managed,
        # cross-platform artifact directory.  File-like objects are preserved.
        from matplotlib.figure import Figure

        original_savefig = getattr(Figure.savefig, "_sevir_original", Figure.savefig)

        def managed_savefig(figure: Any, filename: Any, *args: Any, **kwargs: Any) -> Any:
            if not generate_visual:
                raise PermissionError(
                    "generate_visual is false; image artifact generation is disabled"
                )
            root_text = str(os.getenv("SEVIR_ARTIFACT_DIR", "")).strip()
            if root_text and isinstance(filename, (str, Path)):
                from sevir_world.tools.visualization.renderer import figure_path
                filename = figure_path(filename)
            saved = original_savefig(figure, filename, *args, **kwargs)
            figure._sevir_saved = True
            return saved

        managed_savefig._sevir_original = original_savefig  # type: ignore[attr-defined]
        Figure.savefig = managed_savefig  # type: ignore[method-assign]
    except (ImportError, ValueError):
        return


class WorkerSession:
    """One request-scoped Python namespace shared by every execute step."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.request: ExecutionRequest | None = None
        self.environment: ExecutionEnvironment | None = None
        self.runtime_schema: dict[str, Any] | None = None
        self.namespace: dict[str, Any] | None = None
        self.injected_namespace_names: set[str] = set()
        self.active_tools: set[str] = set()
        self.step = 0
        self.objects = ObjectStore(self.session_id)

    def bind(self, request: ExecutionRequest) -> None:
        active_tools = set(request.selected_tools if request.active_tools is None else request.active_tools)
        if not self.active_tools <= active_tools:
            raise ValueError("tool selections can only grow inside one execution session")
        if self.request is not None:
            if request != self.request:
                raise ValueError(
                    "persistent worker cannot change request binding or selected resources "
                    "inside one execution session"
                )
        else:
            _configure_matplotlib(generate_visual=request.generate_visual)
            self.environment = ExecutionEnvironment(request)
            self.namespace = {
                "__builtins__": restricted_builtins(self.environment.available_modules),
                "objects": self.objects,
                **{name: value for name, value in self.environment.resources.items()
                   if name not in request.selected_tools or name == "ds"},
            }
            self.injected_namespace_names = set(self.namespace)
        self.request = request
        assert self.namespace is not None and self.environment is not None
        for name in active_tools - self.active_tools:
            self.namespace[name] = self.environment.resources[name]
            self.injected_namespace_names.add(name)
        self.active_tools = active_tools
        self.runtime_schema = self.environment.describe()
        for tool in self.runtime_schema["tools"]:
            tool["selected"] = tool["name"] in active_tools


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
        request = ExecutionRequest(
            data_descs=tuple(payload["data_descs"]),
            selected_tools=tuple(payload["selected_tools"]),
            binding=RequestBinding(**payload["binding"]),
            evidence_state=payload.get("evidence_state", {}),
            generate_visual=payload.get("generate_visual", False),
            active_tools=tuple(payload["active_tools"]) if payload.get("active_tools") is not None else None,
        )
        validate_selected_capabilities(request.selected_tools)
        tree = validate_code(code, request.selected_tools)
        phase = "environment_setup"
        session.bind(request)
        environment = session.environment
        runtime_schema = session.runtime_schema
        validate_imports(tree, environment.available_modules)
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
        phase = "state_isolation"
        working_namespace = fork_namespace(namespace, session.injected_namespace_names)
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            phase = "code_compilation"
            program = compile(tree, "<sevir-agent>", "exec")
            phase = "generated_exec"
            exec(program, working_namespace)
            phase = "artifact_capture"
            _capture_open_figures(generate_visual=request.generate_visual)
            phase = "result_serialization"
            result_value = working_namespace.get("result", working_namespace.get("output"))
            result = json_safe(result_value)
            if isinstance(result, dict) and "study_update" in result:
                if not isinstance(result["study_update"], dict):
                    raise TypeError("study_update must be an object")
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
        calls = json_safe(environment.recorder.calls[call_start:])
        successful_calls = sum(
            isinstance(item, dict) and item.get("status") == "ok" for item in calls
        )
        failed_calls = sum(
            not isinstance(item, dict) or item.get("status") != "ok" for item in calls
        )
        # Retention must succeed before publishing variables as committed.
        phase = "state_retention"
        retained_output = session.objects.retain(result_value)
        namespace.clear()
        namespace.update(working_namespace)
        session.step += 1
        observation_quality = "partial" if failed_calls else "complete"
        output = {
            "status": "partial" if failed_calls else "ok",
            "observation_quality": observation_quality,
            "result": result,
            "primary_result": _primary_result(calls, result),
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
            "available_names": _available_namespace_names(
                namespace, session.injected_namespace_names
            ),
            "namespace_state": _namespace_state(
                before_namespace,
                namespace,
                tree,
                committed=True,
                injected_names=session.injected_namespace_names,
            ),
            "worker_session": {"id": session.session_id, "step": session.step},
        }
        output["execution_state"] = state_receipt(session.objects, output["namespace_state"], output=retained_output)
    except Exception as error:
        # A failed transaction must not leak an open Matplotlib figure into
        # the next reflective attempt. Otherwise the later successful step
        # captures both the abandoned and repaired figures and exposes
        # duplicate/stale evidence artifacts.
        try:
            import matplotlib.pyplot as pyplot

            pyplot.close("all")
        except ImportError:
            pass
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
            json_safe(environment.recorder.calls[call_start:]) if environment else []
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
            "error_details": error_record(error, stage=phase),
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
                session.namespace if session is not None else None,
                session.injected_namespace_names if session is not None else set(),
            ),
            "namespace_state": _namespace_state(
                before_namespace,
                working_namespace,
                tree,
                committed=False,
                error_line=(error_location or {}).get("line"),
                injected_names=(
                    session.injected_namespace_names
                    if session is not None else set()
                ),
            ),
            "failed_value_shapes": _failed_value_shapes(
                working_namespace,
                session.namespace if session is not None else None,
                session.injected_namespace_names if session is not None else set(),
            ),
            "runtime_schema": runtime_schema if environment else None,
            "worker_session": {"id": session.session_id, "step": session.step},
        }
        output["execution_state"] = state_receipt(session.objects, output["namespace_state"],
            error=error, tree=tree, transaction="rolled_back" if working_namespace is not None else "not_started")
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
                "error_phase": "output_transport",
                "execution_state": compact_state(output.get("execution_state")),
                "worker_session": output.get("worker_session"),
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
            if payload.get("command") == "execute_file":
                payload = json.loads(Path("request.json").read_text(encoding="utf-8"))
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
