from __future__ import annotations

"""Minimal validation for generated programs.

Execution safety comes from the isolated subprocess, timeout, memory monitor,
temporary working directory and restricted environment owned by ``client.py``.
This module deliberately does not redefine Python syntax or NumPy semantics.
"""

import ast
import builtins
import importlib.util
import re
from typing import Any, Mapping


class UnsafeCodeError(ValueError):
    """The generated source cannot be invoked through the Agent contract."""


SAFE_STDLIB_IMPORT_ROOTS = frozenset({
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "time",
})

# Request data and artifacts enter generated programs through bound objects.
# These builtins would create unrecorded filesystem or dynamic-code side doors.
BLOCKED_BUILTIN_NAMES = frozenset({
    "breakpoint", "compile", "eval", "exec", "input", "open",
})


def allowed_import_roots(available_modules: tuple[str, ...]) -> frozenset[str]:
    """Return the complete import allow-list for generated programs."""

    return frozenset({
        *(str(item).split(".", 1)[0] for item in available_modules),
        *SAFE_STDLIB_IMPORT_ROOTS,
    })


def normalize_generated_code(code: str) -> str:
    text = str(code or "").strip()
    execute = re.search(r"<execute>(.*?)</execute>", text, re.DOTALL | re.IGNORECASE)
    if execute:
        text = execute.group(1).strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text


def validate_code(code: str, selected_tools: tuple[str, ...]) -> ast.Module:
    """Parse ordinary Python source before isolated ``exec(code, ns)``."""

    source = normalize_generated_code(code)
    if not source:
        raise UnsafeCodeError("generated program is empty")
    if len(source) > 100_000:
        raise UnsafeCodeError("generated program exceeds 100,000 characters")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise UnsafeCodeError(f"invalid Python syntax: {error}") from error

    return tree


def validate_imports(tree: ast.Module, available_modules: tuple[str, ...]) -> None:
    """Reject imports outside the declared in-memory analysis environment.

    Merely checking whether a package was installed allowed generated code to
    import ``sevir_world.tools`` directly and bypass request binding, tracing
    and the Capability Gate. The allow-list is now part of the contract.
    """

    available = allowed_import_roots(available_modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            root = name.split(".", 1)[0]
            if root in available:
                continue
            installed = importlib.util.find_spec(root) is not None
            suffix = "installed but not authorised" if installed else "not installed"
            raise UnsafeCodeError(
                f"generated code imports {root!r}, which is {suffix}; "
                "use only the injected request-bound capabilities or approved "
                f"analysis modules: {', '.join(sorted(available))}"
            )


def restricted_builtins(available_modules: tuple[str, ...]) -> Mapping[str, Any]:
    """Return builtins with a runtime import guard matching the AST check."""

    allowed = allowed_import_roots(available_modules)
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        root = str(name or "").split(".", 1)[0]
        if level:
            raise ImportError("relative imports are not available in generated programs")
        if root not in allowed:
            raise ImportError(
                f"import {root!r} is outside the request-bound analysis environment"
            )
        return original_import(name, globals, locals, fromlist, level)

    values = {
        name: value
        for name, value in builtins.__dict__.items()
        if name not in BLOCKED_BUILTIN_NAMES
    }
    values["__import__"] = guarded_import
    return values


def count_external_tool_calls(code: str, selected_tools: tuple[str, ...]) -> int:
    tree = validate_code(code, selected_tools)
    roots = set(selected_tools) - {"ds"}
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value: Any = node.func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name) and value.id in roots:
            count += 1
    return count
