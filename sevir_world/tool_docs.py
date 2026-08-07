from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from .contracts import CANONICAL_ARGUMENTS


POLICY = """Execution rules:
- Define exactly one function named run with the supplied arguments and return JSON-safe data.
- Do not import modules, read files, use the network, launch commands, or access private attributes.
- VIL value 255 is missing ground truth and is excluded from verification metrics.
- VIL thresholds are encoded-value proxies, not rainfall or official warnings.
- Ensemble member support is not a calibrated probability.
- Catalog statistics describe metadata records, not unique disasters.
"""


def _format_arguments(node: ast.FunctionDef) -> str:
    values: list[str] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for argument, default in zip(positional, defaults):
        if argument.arg == "self":
            continue
        text = argument.arg
        if argument.annotation:
            text += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        values.append(text)
    return ", ".join(values)


def extract_function_docstrings(
    path: str | Path, class_name: str
) -> tuple[str, list[dict[str, str]]]:
    """Extract one tool class's public API without importing generated documentation."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods: list[dict[str, str]] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        methods.append(
            {
                "name": node.name,
                "arguments": _format_arguments(node),
                "docstring": ast.get_docstring(node, clean=True) or "No description.",
            }
        )
    return ast.get_docstring(class_node, clean=True) or "", methods


def class_api_docs(tool_name: str, tool_class: type[Any]) -> str:
    """Format AST-extracted public methods as an LLM-facing API manual."""
    module = importlib.import_module(tool_class.__module__)
    path = Path(module.__file__ or "")
    class_doc, methods = extract_function_docstrings(path, tool_class.__name__)
    lines = [f"{tool_name}: {tool_class.__name__}"]
    if class_doc:
        lines.append(class_doc)
    for method in methods:
        lines.append(f"{tool_name}.{method['name']}({method['arguments']})")
        lines.append(f"  {method['docstring']}")
    return "\n".join(lines)


def selected_api_reference(
    selected_tools: tuple[str, ...], registry: dict[str, type[Any]]
) -> str:
    """Generate documentation only for resources selected by the request."""
    sections = [
        "Injected data:\n"
        "ds is a list of dictionaries loaded from data_descs. Each SEVIR item includes "
        "event_id and exact catalog metadata; each upload item includes upload_id and metadata."
    ]
    for name in CANONICAL_ARGUMENTS:
        if name != "ds" and name in selected_tools:
            sections.append(class_api_docs(name, registry[name]))
    sections.append(POLICY.strip())
    return "\n\n".join(sections)
