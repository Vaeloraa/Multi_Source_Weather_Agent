from __future__ import annotations

"""Minimal validation for generated programs.

Execution safety comes from the isolated subprocess, timeout, memory monitor,
temporary working directory and restricted environment owned by ``client.py``.
This module deliberately does not redefine Python syntax or NumPy semantics.
"""

import ast
import importlib.util
import re
from typing import Any


class UnsafeCodeError(ValueError):
    """The generated source cannot be invoked through the Agent contract."""


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
    """Reject missing imports before generated code can call a weather tool."""

    available = set(available_modules)
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
            if importlib.util.find_spec(root) is None:
                raise UnsafeCodeError(
                    f"generated code imports unavailable module {root!r}; "
                    f"available modules: {', '.join(sorted(available))}"
                )


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
