from __future__ import annotations

"""Generate the Agent's capability description from the real tool code."""

import ast
from functools import lru_cache
from pathlib import Path

from .contracts import CANONICAL_ARGUMENTS


POLICY = """Runtime notes:
- Request data, four lazy capability objects and study evidence are injected directly
  into the execution namespace. The selected contract subset only controls prompt size.
  Write ordinary Python and assign compact JSON-safe output to ``result``.
- ``study`` is the read-only structured evidence ledger from earlier requests in
  this session. Use ``study.case_id``, ``study.revision``, ``study.decisions``,
  ``study.records()``, ``study.select(source=None, operation=None, turn_id=None)``,
  ``study.latest(source=None, operation=None)`` and ``study.get(record_id)``.
  These return ordinary dictionaries/tuples containing request facts, real tool
  evidence, derived calculation results and artifacts; prior answer prose is excluded.
- When this analysis makes a durable decision needed by later requests, include
  ``result["study_update"]`` with only applicable fields: ``focus_window``,
  ``candidate_subjects`` (a list of ``subject_key``/``event_id`` dicts),
  ``selected_subject`` (one such dict), ``target_time_utc``,
  ``target_region`` or ``aurora_initialization_time_utc``. This updates case state;
  do not copy explanatory prose or unverified claims into it.
- The same Python namespace and capability instances remain alive across Agent steps.
  Keep useful Python values in variables and refine them without reloading them.
- This is standard Python in an isolated worker. ``np``, ``pd``, ``xr``, ``scipy``,
  ``skimage``, comprehensions, helper functions, normal array methods and
  matplotlib are available; the runtime reports the verified module list.
- Each method below includes its real signature, return type, and a minimal example.
  Together with the source-derived ``RequestBinding`` shown in the capability
  reference, these ordinary Python definitions are the complete contract. Do not
  probe it with dir() or inspect.
- Capability instances already contain the request event, upload, location and time.
  Optional identity arguments are only for deliberately switching subjects.
- Successful methods return the documented standard values directly. Gridded and
  temporal data are xarray ``DataArray``/``Dataset`` objects; catalog and candidate
  tables are pandas ``DataFrame`` objects; small metadata values are flat records.
- The capability value exists in full inside the Worker. If a raw DataArray,
  Dataset or DataFrame is assigned directly to ``result``, the cross-process
  transport returns a typed ``representation=metadata_only`` summary rather than
  its values or rows. Reduce scientific objects to compact JSON-safe evidence in
  the same execution when a later Agent step must read the result.
- ``sevir.search`` returns a DataFrame with canonical ``subject_key``, ``event_id`` and
  ``event_anchor_time_utc`` columns. Its ``observation_status`` describes availability;
  discovery is separate from loading pixels.
- Call ``sevir.observations`` or ``sevir.observation_batch`` to load real observed
  pixels. The loader transparently tries the configured cache and local/remote source.
- Selected tools are available capabilities, not mandatory workflow steps. Choose the
  data, calculations, thresholds and figures that answer the user's actual question.
- Keep large arrays inside the worker and return compact evidence, conclusions and
  artifact references. Never return an entire raster cube through JSON.
- Before arithmetic or plotting, verify that a filtered or thresholded selection is
  not empty; empty arrays silently break broadcasting and statistics. Respect NumPy
  semantics: never use an array directly inside if/and/or (use .any() or .all()),
  and keep operand shapes aligned before elementwise operations.
- When the user asks for a figure, prefer the pre-imported helper
  plot_field(data, title="", cmap="viridis") — it saves the PNG below
  SEVIR_ARTIFACT_DIR and returns its path, plotting the last frame of a 3-D cube.
  Ordinary matplotlib is also available for custom figures; save PNG files below
  os.environ['SEVIR_ARTIFACT_DIR'] and return their paths in ``result``. The runtime
  registers stable artifact URLs automatically. Use default matplotlib styling; do not
  build custom colormaps unless the question explicitly asks for one.
- For data-backed questions, create the evidence figure in the same execution as
  the first successful data call and keep the exact path returned by ``plot_field``
  in ``result``. Never type an ``/api/artifacts/...`` URL yourself.
- Write compact code with no comments, no unused imports and no repeated setup: np and
  math are already available in the execution namespace, so import only the other
  modules you need (for example matplotlib.pyplot).
"""


_MODEL_SOURCE = Path(__file__).resolve().parents[1] / "capabilities" / "models.py"
_BINDING_SOURCE = Path(__file__).resolve().parents[1] / "capabilities" / "base.py"
_TOOL_ROOT = Path(__file__).resolve().parents[1] / "tools"
TOOL_IMPLEMENTATIONS: dict[str, tuple[Path, str]] = {
    "sevir": (_TOOL_ROOT / "sevir" / "Sevir.py", "Sevir"),
    "geolocator": (_TOOL_ROOT / "geolocator" / "Geolocator.py", "Geolocator"),
    "cascast": (_TOOL_ROOT / "cascast" / "CasCast.py", "CasCast"),
    "aurora": (_TOOL_ROOT / "aurora" / "Aurora.py", "Aurora"),
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


def _dataclass_fields(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read public value fields from the actual dataclass definitions.

    This deliberately derives the field list from ``models.py`` instead of
    maintaining a second schema.  The generated method reference therefore
    stays synchronized when a return value changes.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        field_names: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_names.append(item.target.id)
        if field_names:
            result[node.name] = tuple(field_names)
    return result


def _dataclass_api_docs(path: str | Path, class_name: str) -> str:
    """Render one public dataclass and its referenced value types from source."""

    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }

    def default_text(value: ast.expr) -> str:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "field"
        ):
            factory = next(
                (
                    keyword.value for keyword in value.keywords
                    if keyword.arg == "default_factory"
                ),
                None,
            )
            if factory is not None:
                return f"{ast.unparse(factory)}()"
        return ast.unparse(value)

    rendered: list[str] = []
    visited: set[str] = set()

    def render(name: str) -> None:
        if name in visited or name not in classes:
            return
        visited.add(name)
        node = classes[name]
        fields: list[str] = []
        referenced: list[str] = []
        properties: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                text = f"{item.target.id}: {ast.unparse(item.annotation)}"
                if item.value is not None:
                    text += f" = {default_text(item.value)}"
                fields.append(text)
                referenced.extend(
                    child.id for child in ast.walk(item.annotation)
                    if isinstance(child, ast.Name) and child.id in classes
                )
            if (
                isinstance(item, ast.FunctionDef)
                and any(isinstance(decorator, ast.Name) and decorator.id == "property"
                        for decorator in item.decorator_list)
                and item.returns is not None
            ):
                properties.append(f"{item.name}: {ast.unparse(item.returns)}")
                referenced.extend(
                    child.id for child in ast.walk(item.returns)
                    if isinstance(child, ast.Name) and child.id in classes
                )
        lines = [f"{name}({', '.join(fields)})"]
        class_doc = ast.get_docstring(node, clean=True)
        if class_doc:
            lines.append("  " + " ".join(class_doc.split()))
        if properties:
            lines.append("  Properties: " + ", ".join(properties))
        rendered.append("\n".join(lines))
        for referenced_name in dict.fromkeys(referenced):
            render(referenced_name)

    render(class_name)
    return "\n".join(rendered)


def _return_value_names(node: ast.FunctionDef) -> tuple[str, ...]:
    """Return domain-model names mentioned by a return annotation."""
    if node.returns is None:
        return ()
    names = []
    for item in ast.walk(node.returns):
        if isinstance(item, ast.Name) and item.id not in {"Any", "None"}:
            names.append(item.id)
    return tuple(dict.fromkeys(names))


def _value_type_contracts(
    path: str | Path,
    root_names: tuple[str, ...],
) -> str:
    """Recursively render public fields, properties and methods from value source."""

    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    rendered: list[str] = []
    visited: set[str] = set()

    def references(annotation: ast.expr | None) -> list[str]:
        if annotation is None:
            return []
        found = [
            item.id for item in ast.walk(annotation)
            if isinstance(item, ast.Name) and item.id in classes
        ]
        for item in ast.walk(annotation):
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                found.extend(
                    name for name in classes
                    if name in item.value
                )
        return list(dict.fromkeys(found))

    def render(name: str) -> None:
        if name in visited or name not in classes:
            return
        visited.add(name)
        node = classes[name]
        fields: list[str] = []
        properties: list[str] = []
        methods: list[str] = []
        nested: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields.append(f"{name}.{item.target.id}: {ast.unparse(item.annotation)}")
                nested.extend(references(item.annotation))
                continue
            if not isinstance(item, ast.FunctionDef) or item.name.startswith("_"):
                continue
            is_property = any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in item.decorator_list
            )
            return_text = ast.unparse(item.returns) if item.returns is not None else "Any"
            nested.extend(references(item.returns))
            if is_property:
                properties.append(f"{name}.{item.name}: {return_text}")
            else:
                methods.append(
                    f"{name}.{item.name}({_format_arguments(item)}) -> {return_text}"
                )
                for argument in (*item.args.args, *item.args.kwonlyargs):
                    nested.extend(references(argument.annotation))
        parts = []
        if fields:
            parts.append("Fields: " + ", ".join(fields))
        if properties:
            parts.append("Properties: " + ", ".join(properties))
        if methods:
            parts.append("Methods: " + ", ".join(methods))
        if parts:
            rendered.append(". ".join(parts) + ".")
        for nested_name in dict.fromkeys(nested):
            render(nested_name)

    for root_name in root_names:
        render(root_name)
    return " ".join(rendered)


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
                "return_contract": _value_type_contracts(
                    _MODEL_SOURCE,
                    _return_value_names(node),
                ),
            }
        )
    return ast.get_docstring(class_node, clean=True) or "", methods


def class_api_docs(
    tool_name: str,
    source_path: str | Path,
    class_name: str,
) -> str:
    """Render one capability contract directly from its source definition."""

    class_doc, methods = extract_function_docstrings(source_path, class_name)
    lines = [f"{tool_name}: {class_name}"]
    if class_doc:
        lines.append(class_doc)
    for method in methods:
        lines.append(
            f"{tool_name}.{method['name']}({method['arguments']}) -> {method['returns']}"
        )
        docstring = " ".join(method["docstring"].split())
        if method.get("return_contract"):
            docstring += " " + method["return_contract"]
        if docstring:
            lines.append(f"  {docstring}")
    return "\n".join(lines)


def _returns_summary(docstring: str) -> str:
    """Extract the source docstring's Returns section without examples/notes."""

    lines = str(docstring or "").splitlines()
    start = next(
        (
            index + 2
            for index, line in enumerate(lines[:-1])
            if line.strip() == "Returns" and set(lines[index + 1].strip()) == {"-"}
        ),
        None,
    )
    if start is None:
        return ""
    selected: list[str] = []
    for index in range(start, len(lines)):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and set(lines[index + 1].strip()) == {"-"}
        ):
            break
        selected.append(lines[index].strip())
    return " ".join(item for item in selected if item)


def compact_api_reference(selected_tools: tuple[str, ...]) -> str:
    """Render the irreducible repair contract: signatures and returned structure."""

    return _cached_compact_api_reference(tuple(selected_tools))


@lru_cache(maxsize=16)
def _cached_compact_api_reference(selected_tools: tuple[str, ...]) -> str:
    sections: list[str] = []
    for name in CANONICAL_ARGUMENTS:
        if name == "ds" or name not in selected_tools:
            continue
        implementation = TOOL_IMPLEMENTATIONS.get(name)
        if implementation is None:
            continue
        _, methods = extract_function_docstrings(
            implementation[0], implementation[1]
        )
        lines = [f"{name} public core contract (never infer backend arguments):"]
        for method in methods:
            lines.append(
                f"{name}.{method['name']}({method['arguments']}) -> {method['returns']}"
            )
            returned = _returns_summary(method["docstring"])
            if returned:
                lines.append("  Returns: " + returned)
            if method.get("return_contract"):
                lines.append("  " + method["return_contract"])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def selected_api_reference(
    selected_tools: tuple[str, ...],
    *,
    include_policy: bool = True,
) -> str:
    """Render the capability documentation for the selected tools.

    The rendered text depends only on the tool subset, so it is cached per
    subset: repeated requests skip the AST re-parse of every implementation
    module instead of rebuilding identical documentation each time.  The
    docstrings themselves stay untouched.
    """

    return _cached_api_reference(tuple(selected_tools), include_policy)


def capability_index(selected_tools: tuple[str, ...]) -> str:
    """Render a short first-layer index; details are requested capability by capability."""

    return _cached_capability_index(tuple(selected_tools))


@lru_cache(maxsize=16)
def _cached_capability_index(selected_tools: tuple[str, ...]) -> str:
    sections = [
        "Injected values: ds (request-bound records), study (cross-request evidence), "
        "plot_field (quick PNG helper). Four capability objects are lazy.",
        "Each capability exposes the same immutable .request: RequestBinding value. "
        "Its source-derived fields are included in the expanded capability reference.",
        "Before using a capability, request its complete source-derived contract with "
        "<expand>capability_name</expand>. Expand only capabilities needed for the next "
        "analysis; then return one <execute> or <solution> action on a later turn.",
    ]
    for name in CANONICAL_ARGUMENTS:
        if name == "ds" or name not in selected_tools:
            continue
        implementation = TOOL_IMPLEMENTATIONS.get(name)
        if implementation is None:
            continue
        class_doc, methods = extract_function_docstrings(implementation[0], implementation[1])
        summary = class_doc.splitlines()[0].strip() if class_doc else implementation[1]
        signatures = "; ".join(
            f"{name}.{method['name']}({method['arguments']}) -> {method['returns']}"
            for method in methods
        )
        sections.append(f"{name}: {summary}\n  {signatures}")
    return "\n\n".join(sections)


@lru_cache(maxsize=16)
def _cached_api_reference(
    selected_tools: tuple[str, ...], include_policy: bool
) -> str:
    sections = [
        "Injected request data:\n"
        "ds is a Python list of request-bound records, not a tool. Each record identifies "
        "its source and may contain event_id or upload_id, metadata, values (the same array "
        "is also exposed under the key raw_data; they are two separate keys), dimensions, "
        "relative_minutes and missing_value_code. Use normal dict/NumPy operations. The four "
        "capability objects are lazy: merely receiving them imports no model or data backend."
    ]
    sections.append(
        "Request-bound state shared by sevir.request, geolocator.request, "
        "cascast.request and aurora.request:\n"
        + _dataclass_api_docs(_BINDING_SOURCE, "RequestBinding")
    )
    for name in CANONICAL_ARGUMENTS:
        if name == "ds" or name not in selected_tools:
            continue
        implementation = TOOL_IMPLEMENTATIONS.get(name)
        if implementation is None:
            continue
        sections.append(
            class_api_docs(name, implementation[0], implementation[1])
        )
    if include_policy:
        sections.append(POLICY.strip())
    return "\n\n".join(sections)
