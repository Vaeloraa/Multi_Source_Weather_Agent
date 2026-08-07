from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sevir_world.safety import UnsafeCodeError, validate_code  # noqa: E402
from sevir_world.contracts import ExecutionRequest  # noqa: E402
from sevir_world.runtime import ExecutionEnvironment  # noqa: E402


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


def _safe_range(*arguments: int) -> range:
    value = range(*arguments)
    if len(value) > 10_000:
        raise ValueError("range is limited to 10,000 iterations")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"result contains unsupported type: {type(value).__name__}")


def _primary_result(calls: list[dict[str, Any]], result: Any) -> Any:
    successful = [item.get("result") for item in calls if item.get("status") == "ok"]
    dictionaries = [item for item in successful if isinstance(item, dict)]
    if dictionaries:
        return dictionaries[-1]
    return result


def main() -> None:
    output: dict[str, Any]
    environment: ExecutionEnvironment | None = None
    try:
        raw = sys.stdin.read(65_537)
        if len(raw) > 65_536:
            raise ValueError("execution request exceeds 64 KiB")
        payload = json.loads(raw)
        code = str(payload.get("code", ""))
        request = ExecutionRequest.from_payload(
            list(payload.get("data_descs") or []),
            list(payload.get("selected_tools") or []),
        )
        tree = validate_code(code, request.selected_tools)
        environment = ExecutionEnvironment(request)
        captured = BoundedOutput()

        def safe_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
            captured.write(sep.join(str(value) for value in values) + end)

        globals_dict = {
            "__builtins__": {},
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "range": _safe_range,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "abs": abs,
            "all": all,
            "any": any,
            "zip": zip,
            "print": safe_print,
        }
        locals_dict: dict[str, Any] = {}
        exec(compile(tree, "<sevir-direct>", "exec"), globals_dict, locals_dict)
        run_function = locals_dict.get("run")
        if not callable(run_function):
            raise RuntimeError("validated program did not define run")
        result = _json_safe(environment.invoke(run_function))
        calls = _json_safe(environment.recorder.calls)
        output = {
            "status": "ok",
            "result": result,
            "primary_result": _json_safe(_primary_result(calls, result)),
            "calls": calls,
            "stdout": captured.getvalue(),
        }
    except Exception as error:
        output = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception_only(type(error), error)).strip(),
            "calls": _json_safe(environment.recorder.calls) if environment else [],
        }
    # ASCII transport avoids Windows redirected-console code-page ambiguity.
    encoded = json.dumps(output, ensure_ascii=True)
    if len(encoded) > 1_000_000:
        encoded = json.dumps(
            {
                "status": "error",
                "error_type": "OutputLimitError",
                "error": "execution result exceeds 1,000,000 characters",
            },
            ensure_ascii=True,
        )
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
