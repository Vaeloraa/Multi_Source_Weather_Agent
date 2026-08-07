from __future__ import annotations

import ast
from typing import Any

from ..agent.contracts import CANONICAL_ARGUMENTS


SAFE_FUNCTIONS = {
    "len", "min", "max", "sum", "round", "sorted", "enumerate", "range",
    "list", "dict", "tuple", "set", "float", "int", "str", "bool", "abs",
    "all", "any", "zip", "print",
}
SAFE_VALUE_METHODS = {
    "get", "items", "keys", "values", "append", "extend", "count", "index",
    "lower", "upper", "strip", "split", "join", "startswith", "endswith",
}
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.Expr, ast.If, ast.For, ast.Break, ast.Continue, ast.Pass,
    ast.Load, ast.Store, ast.Name, ast.Constant, ast.Dict, ast.List, ast.Tuple,
    ast.Set, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.ListComp,
    ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension, ast.JoinedStr,
    ast.FormattedValue, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
}


class UnsafeCodeError(ValueError):
    """Raised when generated code exceeds the restricted language contract."""


def _attribute_root(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


class CodeValidator(ast.NodeVisitor):
    def __init__(self, selected_tools: tuple[str, ...]) -> None:
        self.selected_tools = selected_tools
        self.parameters = set(selected_tools)
        self.node_count = 0
        self.tool_call_count = 0

    def generic_visit(self, node: ast.AST) -> Any:
        self.node_count += 1
        if self.node_count > 600:
            raise UnsafeCodeError("code exceeds the 600-node AST limit")
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
            if isinstance(target, ast.Name) and target.id in self.parameters:
                raise UnsafeCodeError(f"cannot overwrite injected argument: {target.id}")
            if isinstance(target, ast.Attribute):
                raise UnsafeCodeError("attribute assignment is not allowed")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_FUNCTIONS:
                raise UnsafeCodeError(f"function is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            root = _attribute_root(node.func)
            if root in self.parameters and root != "ds":
                self.tool_call_count += 1
            elif node.func.attr not in SAFE_VALUE_METHODS:
                raise UnsafeCodeError(f"method is not allowed: {node.func.attr}")
        else:
            raise UnsafeCodeError("dynamic call targets are not allowed")
        return self.generic_visit(node)


def validate_code(code: str, selected_tools: tuple[str, ...]) -> ast.Module:
    """Require one top-level run() whose parameters exactly match selected tools."""
    if not isinstance(code, str) or not code.strip():
        raise UnsafeCodeError("code is empty")
    if len(code) > 12_000:
        raise UnsafeCodeError("code exceeds 12,000 characters")
    unknown = set(selected_tools) - set(CANONICAL_ARGUMENTS)
    if unknown:
        raise UnsafeCodeError(f"unknown selected tools: {', '.join(sorted(unknown))}")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise UnsafeCodeError(f"syntax error at line {error.lineno}: {error.msg}") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsafeCodeError("program must contain exactly one top-level run() function")
    run_node = tree.body[0]
    if run_node.name != "run":
        raise UnsafeCodeError("the top-level function must be named run")
    if run_node.decorator_list or run_node.returns or run_node.args.defaults or run_node.args.kw_defaults:
        raise UnsafeCodeError("run decorators, annotations, and default arguments are not allowed")
    if run_node.args.vararg or run_node.args.kwarg or run_node.args.kwonlyargs:
        raise UnsafeCodeError("run variadic and keyword-only arguments are not allowed")
    parameters = tuple(item.arg for item in run_node.args.args)
    if parameters != selected_tools:
        expected = ", ".join(selected_tools)
        raise UnsafeCodeError(f"run parameters must exactly match selected tools: ({expected})")
    if not any(isinstance(node, ast.Return) for node in ast.walk(run_node)):
        raise UnsafeCodeError("run must return the final JSON-serializable analysis")
    if any(isinstance(node, ast.FunctionDef) for node in ast.walk(run_node) if node is not run_node):
        raise UnsafeCodeError("nested functions are not allowed")
    validator = CodeValidator(selected_tools)
    validator.visit(tree)
    if any(name != "ds" for name in selected_tools) and validator.tool_call_count == 0:
        raise UnsafeCodeError("run must call at least one selected tool")
    return tree
