from __future__ import annotations

"""Generate the Agent's capability description from the real tool code."""

import ast
from pathlib import Path

from .contracts import CANONICAL_ARGUMENTS


POLICY = """Runtime notes:
- Selected data and tool names are injected directly into the execution namespace.
  Write ordinary Python and assign compact JSON-safe output to ``result``.
- The same Python namespace and capability instances remain alive across Agent steps.
  Keep useful Python values in variables and refine them without reloading them.
- This is standard Python in an isolated worker. Imports, NumPy, xarray,
  comprehensions, helper functions, normal array methods and matplotlib are available.
- Each method below includes its real signature, return type, and a minimal example;
  together these are the complete contract. Do not probe it with dir() or inspect.
- Capability instances already contain the request event, upload, location and time.
  Optional identity arguments are only for deliberately switching subjects.
- Successful methods return ordinary Python values directly. Use attributes, lists,
  dictionaries and NumPy arrays exactly as documented; there is no result.data wrapper.
- Selected tools are available capabilities, not mandatory workflow steps. Choose the
  data, calculations, thresholds and figures that answer the user's actual question.
- Keep large arrays inside the worker and return compact evidence, conclusions and
  artifact references. Never return an entire raster cube through JSON.
- When the user asks for a figure, use ordinary matplotlib. Save PNG files below
  os.environ['SEVIR_ARTIFACT_DIR'] and return their paths in ``result``. The runtime
  registers stable artifact URLs automatically.
"""


_CAPABILITY_SOURCE = Path(__file__).resolve().parents[1] / "capabilities" / "weather.py"
TOOL_IMPLEMENTATIONS: dict[str, tuple[Path, str]] = {
    "sevir": (_CAPABILITY_SOURCE, "Sevir"),
    "geolocator": (_CAPABILITY_SOURCE, "Geolocator"),
    "cascast": (_CAPABILITY_SOURCE, "CasCast"),
    "aurora": (_CAPABILITY_SOURCE, "Aurora"),
}


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
    if node.args.vararg:
        values.append(f"*{node.args.vararg.arg}")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = argument.arg
        if default is not None:
            text += f" = {ast.unparse(default)}"
        values.append(text)
    if node.args.kwarg:
        values.append(f"**{node.args.kwarg.arg}")
    return ", ".join(values)


def extract_function_docstrings(
    path: str | Path,
    class_name: str,
) -> tuple[str, list[dict[str, str]]]:
    """Read every public method from the implementation itself."""

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
                "returns": ast.unparse(node.returns) if node.returns else "Any",
                "docstring": ast.get_docstring(node, clean=True) or "No description.",
            }
        )
    return ast.get_docstring(class_node, clean=True) or "", methods


def class_api_docs(
    tool_name: str,
    source_path: str | Path,
    class_name: str,
    *,
    doc_limit: int | None = 900,
) -> str:
    """Render one capability contract directly from its source definition."""

    class_doc, methods = extract_function_docstrings(source_path, class_name)
    lines = [f"{tool_name}: {class_name}"]
    if class_doc:
        lines.append(class_doc[:600])
    for method in methods:
        lines.append(
            f"{tool_name}.{method['name']}({method['arguments']}) -> {method['returns']}"
        )
        docstring = " ".join(method["docstring"].split())
        if doc_limit == -1:
            docstring = ""
        elif doc_limit is not None and len(docstring) > doc_limit:
            docstring = docstring[:doc_limit] + " ..."
        if docstring:
            lines.append(f"  {docstring}")
    return "\n".join(lines)


def selected_api_reference(
    selected_tools: tuple[str, ...],
    *,
    include_policy: bool = True,
) -> str:
    sections = [
        "Injected request data:\n"
        "ds is a Python list of request-bound records, not a tool. Each record identifies "
        "its source and may contain event_id/upload_id, metadata, values/raw_data, dimensions, "
        "relative_minutes and missing_value_code. Use normal dict/NumPy operations. The four "
        "capability objects are lazy: merely receiving them imports no model or data backend."
    ]
    for name in CANONICAL_ARGUMENTS:
        if name == "ds" or name not in selected_tools:
            continue
        implementation = TOOL_IMPLEMENTATIONS.get(name)
        if implementation is None:
            continue
        sections.append(
            class_api_docs(name, implementation[0], implementation[1], doc_limit=900)
        )
    if include_policy:
        sections.append(POLICY.strip())
    return "\n\n".join(sections)
