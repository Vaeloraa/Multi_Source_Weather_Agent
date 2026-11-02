"""Recognize pure repeated searches without rewriting generated analysis code."""
import ast
import json


def repeated_search(code, observations):
    """Return prior evidence only when the whole program repeats a known query."""
    try:
        tree = ast.parse(code)
        if len(tree.body) != 1:
            return None
        statement = tree.body[0]
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id not in {"result", "output"}):
            return None
        value = statement.value
        converted = False
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "to_dict"):
            if value.args or len(value.keywords) != 1 or value.keywords[0].arg != "orient":
                return None
            if ast.literal_eval(value.keywords[0].value) != "records":
                return None
            value = value.func.value
            converted = True
        if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute)
                or not isinstance(value.func.value, ast.Name)
                or value.func.value.id != "knowledge" or value.func.attr != "search"):
            return None
        names = ("query", "tools", "limit")
        if len(value.args) > len(names):
            return None
        arguments = dict(zip(names, (ast.literal_eval(arg) for arg in value.args)))
        for keyword in value.keywords:
            if keyword.arg not in names or keyword.arg in arguments:
                return None
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        if "query" not in arguments:
            return None
        arguments = {"tools": None, "limit": 6, **arguments}
        key = json.dumps(arguments, sort_keys=True)
        for observation in observations:
            # Permit the first conversion of a metadata-only table to readable rows.
            if converted and not isinstance(observation.get("result"), list):
                continue
            for call in observation.get("calls", []):
                if (call.get("api") == "knowledge.search" and call.get("status") == "ok"
                        and call.get("retrieval", {}).get("status", "ok") == "ok"
                        and call.get("knowledge_evidence")
                        and json.dumps(call.get("arguments"), sort_keys=True) == key):
                    return call
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    return None
