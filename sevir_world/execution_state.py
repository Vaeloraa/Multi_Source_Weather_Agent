"""Request-local scientific objects, isolated copies and explicit recovery handles.

Only this module defines the state receipt consumed by the Worker and Agent.
Handles are process-local references, never serialized data or durable checkpoints.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields, is_dataclass
from copy import copy
import os
import types

import numpy as np
import pandas as pd
import xarray as xr

STATE_SCHEMA = "execution-state-v1"
OUTPUT_SLOTS = frozenset({"result", "output"})
_FUNCTION_NAMESPACE = object()


def isolated_copy(value, memo=None):
    """Copy scientific values including mutable object cells and metadata.

    pandas deep=True copies the array but deliberately shares Python objects in
    object columns. Those cells need their own copies, using one graph memo.
    xarray copies keep lazy graphs lazy; no .values/.compute is used here.
    """
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, types.ModuleType):
        return value  # Module/service effects are outside the data transaction.
    if isinstance(value, types.FunctionType) and value.__code__.co_filename == "<multi-source-weather-agent>":
        namespace = memo.get(_FUNCTION_NAMESPACE)
        if namespace is None:
            raise TypeError("generated functions cannot be retained as output objects; retain their data instead")
        closure = tuple(types.CellType() for _ in value.__closure__) if value.__closure__ else None
        copied = types.FunctionType(value.__code__, namespace, value.__name__, None, closure)
        memo[id(value)] = copied
        if closure:
            for target, source in zip(closure, value.__closure__):
                try:
                    target.cell_contents = isolated_copy(source.cell_contents, memo)
                except ValueError:  # Empty closure cell remains empty.
                    pass
        copied.__defaults__ = isolated_copy(value.__defaults__, memo)
        copied.__kwdefaults__ = isolated_copy(value.__kwdefaults__, memo)
        copied.__dict__.update(isolated_copy(value.__dict__, memo))
        return copied
    if isinstance(value, np.ndarray):
        base = value.base
        if (isinstance(base, np.ndarray) and not value.dtype.hasobject
                and not base.dtype.hasobject and (base.flags.c_contiguous or base.flags.f_contiguous)):
            copied_base = isolated_copy(base, memo)
            offset = value.__array_interface__["data"][0] - base.__array_interface__["data"][0]
            copied = np.ndarray(value.shape, dtype=value.dtype, buffer=copied_base,
                                offset=offset, strides=value.strides)
        else:
            copied = value.copy(order="K")
        memo[id(value)] = copied
        if value.dtype.hasobject:
            for index in np.ndindex(value.shape):
                copied[index] = isolated_copy(value[index], memo)
        if not value.flags.writeable:
            copied.flags.writeable = False
        return copied
    if isinstance(value, (pd.DataFrame, pd.Series)):
        copied = value.copy(deep=True)
        copied.index = value.index.copy(deep=True)
        if isinstance(value, pd.DataFrame):
            copied.columns = value.columns.copy(deep=True)
        memo[id(value)] = copied
        if isinstance(value, pd.DataFrame):
            for column in range(len(value.columns)):
                if value.dtypes.iloc[column] == object:
                    for row in range(len(value.index)):
                        copied.iat[row, column] = isolated_copy(value.iat[row, column], memo)
        elif value.dtype == object:
            for row in range(len(value.index)):
                copied.iat[row] = isolated_copy(value.iat[row], memo)
        copied.attrs = isolated_copy(value.attrs, memo)
        return copied
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        copied = value.copy(deep=True)
        memo[id(value)] = copied
        copied.attrs = isolated_copy(value.attrs, memo)
        if isinstance(value, xr.Dataset):
            for name in value.variables:
                copied[name].attrs = isolated_copy(value[name].attrs, memo)
        return copied
    if is_dataclass(value) and not isinstance(value, type):
        copied = copy(value)
        memo[id(value)] = copied
        for field in fields(value):
            object.__setattr__(copied, field.name, isolated_copy(getattr(value, field.name), memo))
        return copied
    if isinstance(value, dict):
        copied = {} if type(value) is dict else deepcopy(value, memo)
        memo[id(value)] = copied
        copied.update((isolated_copy(key, memo), isolated_copy(part, memo)) for key, part in value.items())
        return copied
    if isinstance(value, list):
        copied = []
        memo[id(value)] = copied
        copied.extend(isolated_copy(part, memo) for part in value)
        return copied
    if isinstance(value, tuple):
        parts = [isolated_copy(part, memo) for part in value]
        if id(value) in memo:
            return memo[id(value)]
        copied = type(value)(*parts) if hasattr(value, "_fields") else tuple(parts)
        memo[id(value)] = copied
        return copied
    return deepcopy(value, memo)


def fork_namespace(namespace, injected_names):
    """Run against a detached graph; failed writes never reach committed data."""
    working = {name: namespace[name] for name in injected_names if name in namespace}
    working["__builtins__"] = namespace["__builtins__"]
    memo = {id(namespace): working, _FUNCTION_NAMESPACE: working}
    # Trusted injected capability/service objects retain their request identity.
    memo.update((id(namespace[name]), namespace[name]) for name in injected_names if name in namespace)
    for name, value in namespace.items():
        if name in OUTPUT_SLOTS or name in injected_names or name == "__builtins__":
            continue
        working[name] = isolated_copy(value, memo)
    return working


def object_bytes(value, seen=None):
    """Approximate retained memory without materializing lazy grids."""
    import sys
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes + (sum(object_bytes(x, seen) for x in value.flat) if value.dtype == object else 0)
    if isinstance(value, (pd.DataFrame, pd.Series)):
        usage = value.memory_usage(deep=True)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    if isinstance(value, (xr.Dataset, xr.DataArray)):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(object_bytes(k, seen) + object_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set)):
        return sys.getsizeof(value) + sum(object_bytes(x, seen) for x in value)
    return sys.getsizeof(value)


class ObjectStore:
    """Bounded successful output snapshots; reads return isolated copies."""
    def __init__(self, session_id, *, max_items=None, max_bytes=None):
        self.session_id = session_id
        self.max_items = max_items if max_items is not None else int(os.getenv("SEVIR_STATE_MAX_OBJECTS", "8"))
        self.max_bytes = max_bytes if max_bytes is not None else int(os.getenv("SEVIR_STATE_MAX_BYTES", "536870912"))
        self._values = OrderedDict()
        self._sizes = {}
        self._sequence = 0
        self.last_output = None

    def retain(self, value):
        self._sequence += 1
        if value is None:
            self.last_output = None
            return {"retained": False, "reason": "no_output"}
        size = object_bytes(value)
        if size > self.max_bytes or self.max_items < 1:
            self.last_output = None
            return {"retained": False, "reason": "retention_limit", "estimated_bytes": size}
        handle = f"object:{self.session_id}:{self._sequence}"
        snapshot = isolated_copy(value)
        self._values[handle] = snapshot
        self._sizes[handle] = size
        while len(self._values) > self.max_items or sum(self._sizes.values()) > self.max_bytes:
            removed, _ = self._values.popitem(last=False)
            self._sizes.pop(removed)
        self.last_output = handle
        return {"retained": True, "handle": handle, "type": type(value).__name__, "estimated_bytes": size}

    def get(self, handle):
        """Recover an isolated value by its exact receipt handle; never reload data."""
        if handle not in self._values:
            raise KeyError("object handle expired, belongs to another Worker, or was never committed; "
                           "inspect execution_state and reacquire from the original tool/cache")
        return isolated_copy(self._values[handle])

    def describe(self):
        return {"schema_version": STATE_SCHEMA, "scope": "worker_session",
                "session_id": self.session_id, "latest_output": self.last_output,
                "handles": [{"handle": key, "type": type(value).__name__, "estimated_bytes": self._sizes[key]}
                            for key, value in self._values.items()]}


def state_receipt(store, namespace_state, *, error=None, tree=None, transaction="committed", output=None):
    receipt = {**store.describe(), "transaction": transaction,
               "reusable_names": namespace_state.get("safe_to_reuse", []),
               "output_slots": {"names": sorted(OUTPUT_SLOTS), "lifetime": "one_execution"},
               "data_isolation": "detached_copy", "output": output,
               "external_effects_rolled_back": False}
    if isinstance(error, NameError) and getattr(error, "name", None):
        name = error.name
        lines = sorted({node.lineno for node in ast.walk(tree) if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load) and node.id == name}) if tree else []
        recovery = {"name": name, "read_lines": lines, "available": name in receipt["reusable_names"]}
        if name in OUTPUT_SLOTS and store.last_output:
            recovery.update(handle=store.last_output, instruction="Recover the previous successful output with "
                            "objects.get(handle) into a named variable; repair every read before the new output assignment.")
        else:
            recovery["instruction"] = "Use a committed reusable name or an available object handle; do not invent data."
        receipt["missing_dependency"] = recovery
    return receipt


def compact_state(receipt):
    """Small authoritative handoff retained even when scientific rows are offloaded."""
    if not receipt:
        return {}
    return {key: receipt[key] for key in (
        "schema_version", "scope", "session_id", "transaction", "latest_output",
        "output", "missing_dependency", "invalidated", "data_isolation", "reusable_names",
        "reason", "instruction",
    ) if key in receipt}


def invalidated_state(reason):
    return {"schema_version": STATE_SCHEMA, "scope": "worker_session", "transaction": "not_committed",
            "invalidated": True, "reason": str(reason), "reusable_names": [], "handles": [],
            "latest_output": None, "instruction": "Worker data and object handles are unavailable; "
            "reacquire inputs from the original tools/cache in a new Worker. read_context restores evidence only."}
