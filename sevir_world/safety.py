from __future__ import annotations

import ast
from typing import Any


WORLD_CALLS = {
    "world.data.event",
    "world.data.search",
    "world.data.load_hdf5",
    "world.data.cascast_cache",
    "world.data.uploads",
    "world.geolocator.locate",
    "world.geolocator.pixel_to_latlon",
    "world.geolocator.reverse_geocode",
    "world.geolocator.locate_admin",
    "world.forecaster.run",
    "world.forecaster.hazards",
    "world.forecaster.evaluate_dataset",
    "world.forecaster.uploaded_sequence",
    "world.statistics.historical",
    "world.statistics.compare_event",
    "world.statistics.compare_storm_types",
    "world.uncertainty.analyze",
    "world.uncertainty.track_objects",
    "world.uncertainty.match_ensemble_objects",
    "world.metrics.evaluate_event",
    "world.metrics.evaluate_dataset",
    "world.visualizer.lead_time_metrics",
    "world.visualizer.storm_tracks",
    "world.visualizer.storm_type_comparison",
    "world.visualizer.ensemble_consensus",
}

SAFE_FUNCTIONS = {
    "len",
    "min",
    "max",
    "sum",
    "round",
    "sorted",
    "enumerate",
    "range",
    "list",
    "dict",
    "tuple",
    "set",
    "float",
    "int",
    "str",
    "bool",
    "abs",
    "all",
    "any",
    "zip",
    "print",
}

SAFE_VALUE_METHODS = {
    "get",
    "items",
    "keys",
    "values",
    "append",
    "extend",
    "count",
    "index",
    "lower",
    "upper",
    "strip",
    "split",
    "join",
    "startswith",
    "endswith",
}

ALLOWED_NODES = {
    ast.Module,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.If,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Load,
    ast.Store,
    ast.Name,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}


class UnsafeCodeError(ValueError):
    """Raised when generated code exceeds the restricted language contract."""


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class CodeValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.node_count = 0
        self.world_call_count = 0

    def generic_visit(self, node: ast.AST) -> Any:
        self.node_count += 1
        if self.node_count > 500:
            raise UnsafeCodeError("code exceeds the 500-node AST limit")
        if type(node) not in ALLOWED_NODES:
            raise UnsafeCodeError(f"syntax is not allowed: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("_"):
            raise UnsafeCodeError("private and dunder names are not allowed")
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_"):
            raise UnsafeCodeError("private and dunder attributes are not allowed")
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"world", "context"}:
                raise UnsafeCodeError(f"cannot overwrite {target.id}")
            if isinstance(target, ast.Attribute):
                raise UnsafeCodeError("attribute assignment is not allowed")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_FUNCTIONS:
                raise UnsafeCodeError(f"function is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            path = _attribute_path(node.func)
            if path in WORLD_CALLS:
                self.world_call_count += 1
            elif node.func.attr not in SAFE_VALUE_METHODS:
                raise UnsafeCodeError(f"method is not allowed: {path or node.func.attr}")
        else:
            raise UnsafeCodeError("dynamic call targets are not allowed")
        return self.generic_visit(node)


def validate_code(code: str) -> ast.Module:
    """Parse and validate code against the restricted SEVIRWorld language."""
    if not isinstance(code, str) or not code.strip():
        raise UnsafeCodeError("code is empty")
    if len(code) > 12_000:
        raise UnsafeCodeError("code exceeds 12,000 characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise UnsafeCodeError(f"syntax error at line {error.lineno}: {error.msg}") from error
    validator = CodeValidator()
    validator.visit(tree)
    if validator.world_call_count == 0:
        raise UnsafeCodeError("code must call at least one approved world API")
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if "result" not in assigned_names:
        raise UnsafeCodeError("code must assign the final analysis to result")
    return tree
