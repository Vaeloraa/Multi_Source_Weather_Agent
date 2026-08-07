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
from sevir_world.world import SevirWorld  # noqa: E402


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


def main() -> None:
    output: dict[str, Any]
    try:
        raw = sys.stdin.read(65_537)
        if len(raw) > 65_536:
            raise ValueError("execution request exceeds 64 KiB")
        payload = json.loads(raw)
        code = str(payload.get("code", ""))
        context = dict(payload.get("context") or {})
        tree = validate_code(code)
        world = SevirWorld(context)
        captured = BoundedOutput()

        def safe_print(*values: Any, sep: str = " ", end: str = "\n") -> None:
            captured.write(sep.join(str(value) for value in values) + end)

        globals_dict = {
            "__builtins__": {},
            "world": world,
            "context": context,
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
        result = _json_safe(locals_dict.get("result"))
        output = {
            "status": "ok",
            "result": result,
            "primary_result": _json_safe(world.primary_result()),
            "calls": _json_safe(world.call_trace()),
            "stdout": captured.getvalue(),
        }
    except Exception as error:
        output = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception_only(type(error), error)).strip(),
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
