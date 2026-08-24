from __future__ import annotations

"""Generate the Agent's capability description from the real tool code."""

import ast
from pathlib import Path
from typing import Any

from .contracts import CANONICAL_ARGUMENTS


POLICY = """Runtime notes:
- Selected data and tool names are injected directly into the execution namespace.
  Write ordinary Python and assign compact JSON-safe output to ``result``. A
  synchronous ``run(...)`` function remains accepted for compatibility but is optional.
- After a successful step, its JSON-safe ``result`` is injected into the next
  execution, so follow-up code can refine it without recomputing successful work.
- This is standard Python in an isolated worker. Imports, NumPy, xarray,
  comprehensions, helper functions, normal array methods and matplotlib are available.
- The complete public method names and signatures are listed below. Call only those
  methods and do not spend an execution step rediscovering the interface.
- Event methods with event_id=None use the request-bound event. Upload methods with
  upload_id=None use the request-bound upload. Pass an explicit identity only when the
  question deliberately switches to another event or upload.
- Tool dictionaries use one stable read root: result.data. Status is available through
  result.status/result.result_status. Raw arrays stay in the Worker.
- Selected tools are available capabilities, not mandatory workflow steps. Choose the
  data, calculations, thresholds and figures that answer the user's actual question.
- Keep large arrays inside the worker and return compact evidence, conclusions and
  artifact references. Never return an entire raster cube through JSON.
- When the user asks for a figure, use ordinary matplotlib. Save PNG files below
  os.environ['SEVIR_ARTIFACT_DIR'] and return their paths in ``result``. The runtime
  registers stable artifact URLs automatically.
"""


_CAPABILITY_SOURCE = Path(__file__).resolve().parents[1] / "tools" / "capabilities.py"
TOOL_IMPLEMENTATIONS: dict[str, tuple[Path, str]] = {
    "sevir": (_CAPABILITY_SOURCE, "SevirCapability"),
    "geolocator": (_CAPABILITY_SOURCE, "GeolocatorCapability"),
    "cascast": (_CAPABILITY_SOURCE, "CasCastCapability"),
    "aurora": (_CAPABILITY_SOURCE, "AuroraCapability"),
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
    allowed_methods: tuple[str, ...] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Read every public method from the implementation itself.

    ``allowed_methods`` is accepted for source compatibility and intentionally
    ignored: documentation must describe the real selected object, not a second
    method-level permission protocol.
    """

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
        lines.append(f"{tool_name}.{method['name']}({method['arguments']})")
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
    registry: dict[str, type[Any]] | None = None,
    dataset_schema: dict[str, Any] | None = None,
    *,
    compact: bool = False,
    include_policy: bool = True,
    method_map: dict[str, tuple[str, ...]] | None = None,
) -> str:
    sections = [
        "Injected request data:\n"
        "ds is a Python list of request-bound records, not a tool. Each record identifies "
        "its source and may contain event_id/upload_id, metadata, values/raw_data, dimensions, "
        "relative_minutes and missing_value_code. Use normal dict/NumPy operations. The four "
        "capability objects are lazy: merely receiving them imports no model or data backend."
    ]
    if dataset_schema:
        sections.append(
            "Actual runtime dataset description (loaded before generation):\n"
            + repr(dataset_schema)
        )
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
