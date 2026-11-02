"""Lightweight contracts shared by discovery, execution and result provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
import inspect
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping


CONTRACT_VERSION = "capability-contract-v1"
PRODUCT_VERSION = "product-metadata-v1"
PRODUCT_ROLES = frozenset({
    "forecast", "observation", "ground_truth", "reanalysis", "unavailable", "mixed",
})


class ArgumentValidationError(ValueError):
    """An invalid public call rejected before entering the implementation."""

    code = "ARGUMENT_INVALID"
    phase = "input_validation"
    backend_executed = False

    def __init__(self, tool: str, issues: list[dict[str, Any]]) -> None:
        self.tool = tool
        self.issues = issues
        self.details = {"tool": tool, "contract_version": CONTRACT_VERSION}
        super().__init__(f"{tool}: invalid arguments: {issues}")


@dataclass(frozen=True)
class ParameterSpec:
    """Constraints that a Python signature cannot express on its own."""

    allowed: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


def tool_parameters(**rules: ParameterSpec):
    """Declare constraints beside a method; do not wrap or execute that method."""
    def declare(method):
        unknown = set(rules) - set(inspect.signature(method).parameters)
        if unknown:
            raise ValueError(f"unknown parameter declarations: {sorted(unknown)}")
        method.__tool_parameters__ = MappingProxyType(dict(rules))
        return method
    return declare


@dataclass(frozen=True)
class ExecutionRequirements:
    """A deployment-bounded worker profile, fixed for a selected capability set."""

    profile: str = "standard"
    environment: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """Additional semantics for one real public Python method."""

    name: str
    display_name: str = ""
    stage: str = "analysis"
    result_kind: str = "data"
    model_id: str = ""
    cost: float = 1.0
    batch_parameters: tuple[str, ...] = ()

    def validate(self, method: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(method)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as error:
            raise ArgumentValidationError(self.name, [{"rule": "signature", "message": str(error)}]) from error
        bound.apply_defaults()
        issues = []
        primitives = {"str": str, "bool": bool, "int": Integral, "float": Real}
        invalid_types: set[str] = set()
        for name, value in bound.arguments.items():
            annotation = signature.parameters[name].annotation
            annotation = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
            if value is None and "None" in {part.strip() for part in annotation.split("|")}:
                continue
            expected = primitives.get(annotation.replace(" | None", "").strip())
            if expected is not None and (
                not isinstance(value, expected)
                or (expected in {Integral, Real} and isinstance(value, bool))
            ):
                invalid_types.add(name)
                issues.append({"parameter": name, "rule": "type", "received_type": type(value).__name__, "expected": annotation})
        for name, rule in getattr(method, "__tool_parameters__", {}).items():
            if name in invalid_types:
                continue
            value = bound.arguments[name]
            if rule.allowed and value not in rule.allowed:
                issues.append({"parameter": name, "rule": "enum", "received": value, "allowed": list(rule.allowed)})
            if rule.minimum is not None or rule.maximum is not None:
                valid = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                if not valid or (rule.minimum is not None and value < rule.minimum) or (rule.maximum is not None and value > rule.maximum):
                    issues.append({"parameter": name, "rule": "range", "received": value, "minimum": rule.minimum, "maximum": rule.maximum})
        if issues:
            raise ArgumentValidationError(self.name, issues)
        return dict(bound.arguments)

    def estimate_cost(self, arguments: Mapping[str, Any], cache_hit: bool) -> float:
        cost = self.cost
        if self.batch_parameters:
            count = next((len(arguments[key]) for key in self.batch_parameters if arguments.get(key)), 1)
            cost += min(max(count, 1), 20) * 0.75
        return round(min(cost, 0.25) if cache_hit else cost, 2)


@dataclass(frozen=True)
class CapabilitySpec:
    """Trusted registration; importing this record never loads a model."""

    name: str
    module: str
    class_name: str
    display_name: str
    operations: tuple[ToolSpec, ...] = ()
    execution: ExecutionRequirements = field(default_factory=ExecutionRequirements)
    notes: str = ""
    readiness: str = ""
    source_path: str = ""
    default_enabled: bool = False


@dataclass(frozen=True)
class ProductMetadata:
    """Adapter-declared product facts, captured before generated code sees data."""

    model_id: str
    model_version: str = ""
    input_identity: Mapping[str, Any] = field(default_factory=dict)
    initialization_time_utc: str = ""
    variables: Mapping[str, Any] = field(default_factory=dict)
    segments: tuple[dict[str, Any], ...] = ()
    cache_identity: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": PRODUCT_VERSION, **asdict(self)}


def capture_product(operation: ToolSpec, value: Any, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate adapter metadata and detach it from mutable scientific objects."""
    if operation.result_kind != "model_product":
        return None
    metadata = getattr(value, "attrs", {}).get("product_metadata")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != PRODUCT_VERSION:
        raise ValueError(f"{operation.name} must return adapter-declared {PRODUCT_VERSION}")
    product = deepcopy(metadata)
    if not product.get("model_id") or (operation.model_id and product["model_id"] != operation.model_id):
        raise ValueError(f"{operation.name} returned inconsistent model identity")
    variables = product.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise ValueError("model product must declare variable roles")
    actual = getattr(value, "data_vars", None)
    if actual is not None:
        if not actual:
            raise ValueError("model product contains no variables")
        if set(actual) - set(variables):
            raise ValueError("model product contains variables without declared roles")
        product["variables"] = {name: variables[name] for name in actual}
    for name, variable in product["variables"].items():
        if not isinstance(variable, dict) or variable.get("role") not in PRODUCT_ROLES:
            raise ValueError(f"invalid product role for {name}")
        if actual is not None:
            field_value = actual[name]
            if list(variable.get("dimensions", [])) != list(field_value.dims):
                raise ValueError(f"inconsistent product dimensions for {name}")
            if variable.get("units", "") != field_value.attrs.get("units", ""):
                raise ValueError(f"inconsistent product units for {name}")
    for segment in product.get("segments", ()):
        if not isinstance(segment, dict) or segment.get("role") not in PRODUCT_ROLES:
            raise ValueError("invalid product segment role")
        coordinate = segment.get("coordinate")
        if coordinate not in getattr(value, "coords", {}) or segment.get("value") not in value.coords[coordinate].values:
            raise ValueError("product segment is absent from returned coordinates")
    product["operation"] = operation.name
    product["product_id"] = hashlib.sha256(json.dumps({
        "operation": operation.name, "arguments": arguments, "product": product,
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
    # Fail at the boundary for non-transportable metadata, rather than dropping it later.
    return json.loads(json.dumps(product, ensure_ascii=True))
