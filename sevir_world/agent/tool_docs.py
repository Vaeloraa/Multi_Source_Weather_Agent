from __future__ import annotations

"""Generate the Agent's capability description from the real tool code."""

import ast
from functools import lru_cache
from pathlib import Path

from ..capabilities.registry import RegistryView, capability_names, get_capability, registry_revision
from ..capabilities.dataframe_schema import PUBLIC_DATAFRAME_SCHEMAS, SCHEMAS
from ..capabilities.data_contracts import (
    METHOD_DATA_CONTRACTS, REGION_TYPES, contract_source_paths, render_data_contract,
)


POLICY = """Runtime notes:
- Request data, Agent-selected lazy capability objects and study evidence are injected
  into the execution namespace. Only selected capabilities can be used. Select another
  capability before using it; adding a selection preserves existing Python variables.
  Write ordinary Python and assign compact JSON-safe output to ``result``.
- ``plot_field`` is injected only when the request's structured
  ``generate_visual`` value is true. When it is false, image creation is disabled.
  ``plot_style(name)`` is available with plotting; it resolves a named style only when called.
- ``study`` is the read-only structured evidence ledger from earlier requests in
  this session. Use ``study.case_id``, ``study.revision``, ``study.decisions``,
  ``study.model_runs`` (actual model runs, product identities and failures),
  ``study.records()``, ``study.select(source=None, operation=None, turn_id=None)``,
  ``study.latest(source=None, operation=None)`` and ``study.get(record_id)``.
- ``case_memory`` is an explicit, evidence-only cross-session index. Use
  ``case_memory.search(query='', event_id='', source='', operation='',
  include_current_case=False, limit=5)`` only when the user asks about a prior
  case or when a route explicitly requires prior-case evidence. It returns
  session/turn/record provenance, never old answer prose, and never changes the
  current request binding.
  These return ordinary dictionaries/tuples containing request facts, real tool
  evidence, derived calculation results and artifacts; prior answer prose is excluded.
- ``analytics`` is the deterministic, versioned analysis layer for values already
  authorized in this Worker. ``analytics.describe()`` returns operator/profile schemas.
  ``analytics.execute(request, datasets)`` accepts an ``analysis-request-v1`` mapping and a
  mapping from its declared ``data_handle`` values to Worker-local arrays, and returns
  an ``AnalysisResult`` with ``status``, ``values``, ``sample``, ``provenance`` and
  ``limitations``. It cannot acquire data or widen the selected capability allow-list.
- When this analysis makes a durable decision needed by later requests, include
  an explicit JSON object in ``result["study_update"]``. Its fields are merged
  into case decisions without inference from other result fields or tool names.
  Publish complete comparison tables and use descriptive decision names.
  For asset continuation, publish ``selected_subject`` as a dict containing
  ``subject_key`` and ``event_id``. A null decision removes that field; an empty
  list replaces a prior list. Do not publish explanatory prose or unverified claims.
- The same Python namespace and capability instances remain alive across Agent steps.
  Keep useful Python values in variables and refine them without reloading them.
  Failed executions roll back namespace assignments. ``namespace_state.safe_to_reuse``
  lists retained values; recreate uncommitted and unreached assignments. Diagnostic
  ``failed_value_shapes`` describes rolled-back values, not reusable objects.
  The ``result`` slot is per execution and is never inherited.
- ``execution_state`` is the authoritative Worker state receipt. Successful outputs
  retained within its limits have an exact ``object:SESSION:VERSION`` handle;
  recover one with ``events = objects.get(handle)``. The returned value is an
  isolated copy. ``objects.describe()`` lists live handles. Handles expire on
  eviction or Worker restart and never cross requests. Read_context reads JSON
  evidence; it cannot restore a Python object. Always publish a new result.
- This is standard Python in an isolated worker. ``np``, ``pd``, ``xr``, ``scipy``,
  ``skimage``, comprehensions, helper functions, normal array methods and
  matplotlib are available; the runtime reports the verified module list.
- Each method includes its real signature and critical Data contract. Complete
  parameters and examples are in its tool:CAPABILITY.METHOD archive. Together
  with the source-derived ``RequestBinding``, these are the tool contract. Do not
  probe it with dir() or inspect.
- Capability instances contain only runtime asset identities and the data-access policy.
  Time, location, region, variables and forecast windows must be passed explicitly to
  the relevant public method; never infer them from ``.request``.
- Successful methods return the documented standard values directly. Gridded and
  temporal data are xarray ``DataArray``/``Dataset`` objects; catalog and candidate
  tables are pandas ``DataFrame`` objects; small metadata values are flat records.
  Every public DataFrame carries ``dataframe_schema_id``,
  ``dataframe_schema_version`` and ``dataframe_contract_version`` in ``attrs``.
  The generated Data contract and DataFrame schemas are authoritative if a prose
  docstring gives only a shorter summary. Use only its canonical columns and
  never invent a column name. ``event_type`` is the weather-event label,
  ``channel`` is the physical/product channel, and source aliases such as
  ``img_type`` or ``data_max`` are never public columns.
- The capability value exists in full inside the Worker. If a raw DataArray,
  Dataset or DataFrame is assigned directly to ``result``, the cross-process
  transport returns a typed ``representation=metadata_only`` summary rather than
  its values or rows. Reduce scientific objects to compact JSON-safe evidence in
  the same execution when a later Agent step must read the result.
- Selected tools are available capabilities, not mandatory workflow steps. Choose the
  data, calculations, thresholds and figures that answer the user's actual question.
- Discovery describes available products; it does not execute a model. Cache misses
  are distinct from unsupported parameters or unavailable live inference.
- Keep large arrays inside the worker and return compact evidence, conclusions and
  artifact references. Never return an entire raster cube through JSON.
- Before arithmetic or plotting, verify that a filtered or thresholded selection is
  not empty; empty arrays silently break broadcasting and statistics. Respect NumPy
  semantics: never use an array directly inside if/and/or (use .any() or .all()),
  and keep operand shapes aligned before elementwise operations.
- When the structured ``generate_visual`` contract is true, prefer the pre-imported helper
  plot_field(data, title="", cmap="viridis") — it saves the PNG below. Choose the
  style explicitly from variable metadata and the selected capability notes.
  The helper writes to the managed artifact directory and
  returns its path, plotting the last frame of a
  3-D cube. Ordinary matplotlib is also available for custom figures; leave the
  completed figure open and the Worker captures it into the managed directory.
  Generated code has no direct filesystem/process access. The runtime registers stable
  artifact URLs automatically. Never call ``savefig`` with ``/tmp``, ``./`` or another
  chosen path and never import ``os`` or ``io`` for artifact handling; leave custom
  matplotlib figures open for managed capture. Use default matplotlib styling; do not build custom
  colormaps unless the requested visualization requires one.
- When ``generate_visual`` is true for a data-backed question, create the evidence figure in the same execution as
  the first successful data call and keep the exact path returned by ``plot_field``
  in ``result``. Never type an ``/api/artifacts/...`` URL yourself.
- Write compact code with no comments, no unused imports and no repeated setup: np and
  math are already available in the execution namespace, so import only the other
  modules you need (for example matplotlib.pyplot).
"""


_MODEL_SOURCE = Path(__file__).resolve().parents[1] / "capabilities" / "models.py"
_BINDING_SOURCE = Path(__file__).resolve().parents[1] / "capabilities" / "base.py"
_TOOL_ROOT = Path(__file__).resolve().parents[1] / "tools"
_GEOSPATIAL_MODEL_SOURCE = _TOOL_ROOT / "geospatial" / "contracts.py"
TOOL_IMPLEMENTATIONS = RegistryView(sources=True)


def capability_catalog(available_tools: tuple[str, ...]) -> str:
    """Describe choices without expanding method signatures or loading tools."""
    lines = ["Available capability catalog (names and purposes only):",
             "- ds: request-bound data handles; always available."]
    for name in available_tools:
        if name != "ds":
            spec = get_capability(name)
            lines.append(f"- {name}: {spec.display_name}")
    lines.append(
        'Before executing Python, select only needed capabilities with '
        '<select_tools>["capability_name"]</select_tools>, using names listed above. '
        '[] selects only ds '
        'and the common Python/study environment. The next message supplies '
        'the selected interfaces. Later select_tools actions add capabilities '
        'without discarding previous selections or Python values. '
        'For a clarification or an answer requiring no tools, return a solution directly.'
    )
    return "\n".join(lines)


def dataframe_schema_reference(selected_tools: tuple[str, ...]) -> str:
    """Render exact public table columns from the authoritative registry."""

    selected = set(selected_tools)
    lines = [
        "Authoritative public DataFrame schemas (use these exact columns; do not guess):"
    ]
    for boundary, schema_id in sorted(PUBLIC_DATAFRAME_SCHEMAS.items()):
        capability = boundary.split(".", 1)[0]
        if capability not in selected:
            continue
        schema = SCHEMAS[schema_id]
        columns = ", ".join(item.name for item in schema.columns)
        lines.append(
            f"- {boundary} -> {schema.id}@{schema.version}: [{columns}]"
        )
    return "\n".join(lines) if len(lines) > 1 else ""


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
    elif node.args.kwonlyargs:
        values.append("*")
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = argument.arg
        if argument.annotation:
            text += f": {ast.unparse(argument.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        values.append(text)
    if node.args.kwarg:
        values.append(f"**{node.args.kwarg.arg}")
    return ", ".join(values)


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


def _parameter_docs(method: ast.FunctionDef, tree: ast.Module, path: Path) -> str:
    """Read declarations and literal constants without importing provider code."""
    declarations = [node for node in method.decorator_list if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "tool_parameters"]
    if not declarations:
        return ""
    lines = [ast.unparse(node) for node in declarations]
    names = {node.id for declaration in declarations for node in ast.walk(declaration)
             if isinstance(node, ast.Name)} - {"tool_parameters", "ParameterSpec"}
    for name in sorted(names):
        source, target = tree, name
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
                continue
            imported = next((part for part in node.names if (part.asname or part.name) == name), None)
            if imported is not None:
                source_path = path.parents[node.level - 1] / (node.module.replace(".", "/") + ".py")
                source = ast.parse(source_path.read_text(encoding="utf-8"))
                target = imported.name
                break
        for node in source.body:
            if isinstance(node, ast.Assign) and any(isinstance(part, ast.Name) and part.id == target for part in node.targets):
                lines.append(f"{name}={ast.unparse(node.value)}")
    return "; ".join(lines)


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
                "constraints": _parameter_docs(node, tree, path),
                "returns": ast.unparse(node.returns) if node.returns else "Any",
                "docstring": ast.get_docstring(node, clean=True) or "No description.",
                "return_contract": " ".join(filter(None, (
                    _value_type_contracts(_MODEL_SOURCE, _return_value_names(node)),
                    _value_type_contracts(_GEOSPATIAL_MODEL_SOURCE, _return_value_names(node)),
                ))),
            }
        )
    return ast.get_docstring(class_node, clean=True) or "", methods


def class_api_docs(
    tool_name: str,
    source_path: str | Path,
    class_name: str,
    *, compact: bool = False,
) -> str:
    """Render one capability contract directly from its source definition."""

    class_doc, methods = extract_function_docstrings(source_path, class_name)
    lines = [f"{tool_name}: {class_name}"]
    if class_doc:
        lines.append(class_doc.split("\n\n", 1)[0] if compact else class_doc)
    lines.append(f"{tool_name}.request -> RequestBinding")
    for method in methods:
        lines.append(
            f"{tool_name}.{method['name']}({method['arguments']}) -> {method['returns']}"
        )
        if method["constraints"]:
            lines.append("Declared constraints: " + method["constraints"])
        contract = render_data_contract(f"{tool_name}.{method['name']}", compact=compact)
        if contract:
            lines.append(contract)
        docstring = method["docstring"]
        if docstring and (not compact or not contract):
            lines.append(docstring.split("\n\n", 1)[0] if compact else docstring)
        if compact:
            lines.append(f"Full contract/examples: tool:{tool_name}.{method['name']}")
        elif method.get("return_contract"):
            lines.append(method["return_contract"])
        lines.append("")
    return "\n".join(lines)


def tool_contract_archive(selected_tools: tuple[str, ...]) -> dict[str, str]:
    """Keep full method contracts recoverable without pinning them in every prompt."""
    contracts: dict[str, str] = {}
    for name in selected_tools:
        if name == "ds":
            continue
        path, class_name = TOOL_IMPLEMENTATIONS[name]
        class_doc, methods = extract_function_docstrings(path, class_name)
        notes = get_capability(name).notes
        for method in methods:
            text = [f"{name}.{method['name']}({method['arguments']}) -> {method['returns']}",
                    render_data_contract(f"{name}.{method['name']}"),
                    class_doc, method["docstring"], method["return_contract"], notes]
            if method["constraints"]:
                text.append("Declared constraints: " + method["constraints"])
            contracts[f"tool:{name}.{method['name']}"] = "\n\n".join(item for item in text if item)
    return contracts


def selected_api_reference(
    selected_tools: tuple[str, ...],
    *,
    include_policy: bool = True,
    compact: bool = False,
) -> str:
    """Render the capability documentation for the selected tools.

    Cache by tool subset, registry revision and source file versions. Changes
    to a selected tool's docstrings invalidate its rendered reference without
    requiring a process restart; unchanged references skip repeated AST reads.
    """

    paths = {_BINDING_SOURCE, _MODEL_SOURCE, _GEOSPATIAL_MODEL_SOURCE, Path(__file__)}
    paths.update(contract_source_paths(selected_tools))
    paths.update(Path(TOOL_IMPLEMENTATIONS[name][0]) for name in selected_tools if name != "ds")
    source_revision = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                            for path in sorted(paths))
    return _cached_api_reference(tuple(selected_tools), include_policy, registry_revision(), source_revision, compact)


@lru_cache(maxsize=16)
def _cached_api_reference(
    selected_tools: tuple[str, ...], include_policy: bool, revision: int = 0,
    source_revision: tuple = (),
    compact: bool = False,
) -> str:
    sections = [
        "Injected request data:\n"
        "ds is an indexable, lazy collection of request-bound records, not a tool. "
        "Indexing ds loads that record once; len(ds) and environment inspection load no data. Each record identifies "
        "its source and may contain event_id or upload_id, metadata, values (the same array "
        "is also exposed under the key raw_data; they are two separate keys), dimensions, "
        "relative_minutes and missing_value_code. Use normal dict/NumPy operations. "
        "Record variables, shapes and spatial metadata depend on its source: inspect that record's "
        "dimensions/metadata before indexing; ds itself has no universal xarray coordinates. "
        "Selected capability objects are lazy: merely receiving them imports no model or data backend."
    ]
    sections.append(
        "Request-bound state shared by every selected capability's .request value:\n"
        + _dataclass_api_docs(_BINDING_SOURCE, "RequestBinding")
        + "\n"
        + _dataclass_api_docs(_GEOSPATIAL_MODEL_SOURCE, "BoundingBox")
    )
    used_regions = {contract.region for operation, contract in METHOD_DATA_CONTRACTS.items()
                    if operation.split(".")[0] in selected_tools and contract.region != "none"}
    if used_regions:
        sections.append("Region input types (method-specific; not interchangeable):\n" + "\n".join(
            f"- {name}: {REGION_TYPES[name]}" for name in sorted(used_regions)
        ))
    for name in capability_names():
        if name == "ds" or name not in selected_tools:
            continue
        implementation = TOOL_IMPLEMENTATIONS[name]
        sections.append(
            class_api_docs(name, implementation[0], implementation[1], compact=compact)
        )
        spec = get_capability(name)
        if spec.notes:
            sections.append(f"{name} capability notes:\n{spec.notes}")
        for operation in spec.operations:
            sections.append(
                f"{operation.name}: result_kind={operation.result_kind}; "
                f"model_id={operation.model_id or 'not_applicable'}"
            )
    if include_policy:
        sections.append(POLICY.strip())
    if compact:
        sections.append(
            'The method list is a compact index, not its complete argument/return contract. '
            'Before using an unfamiliar method or guessing result fields, read its exact contract with '
            '<read_context>{"id":"tool:CAPABILITY.METHOD","offset":0,"limit":4000}</read_context>. '
            'Follow next_offset for remaining parameters/examples. Reading a contract does not execute '
            'the tool, load data or spend tool-cost units. Only selected tools have contract archives.'
        )
    return "\n\n".join(sections)
