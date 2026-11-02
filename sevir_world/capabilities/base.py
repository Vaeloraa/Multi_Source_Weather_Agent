from __future__ import annotations

"""Public capability invocation, budgets, request binding and result conversion."""

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from sevir_world.errors import classify_failure, error_record
from ..serialization import SummaryLimits, serialize_value
from .contracts import capture_product
from .registry import operation_spec, operation_display
from ..execution_state import isolated_copy


T = TypeVar("T")

CALL_ARGUMENT_LIMITS = SummaryLimits(max_characters=20_000)
CALL_RESULT_LIMITS = SummaryLimits(max_characters=40_000)


@dataclass(frozen=True, slots=True)
class RequestAssets:
    """Asset identities selected for one request."""

    event_id: str = ""
    upload_id: str = ""
    sevir_subject_key: str = ""


@dataclass(frozen=True, slots=True)
class DataAccessPolicy:
    """Whether capability backends may acquire live data or use cache only."""

    mode: str = "cached"


@dataclass(frozen=True, slots=True)
class RequestBinding:
    """Immutable runtime asset identities and data-access policy.

    Scientific time, place, region and forecast parameters are deliberately
    absent. The Agent must pass those semantics explicitly to each tool call.
    """

    event_id: str = ""
    upload_id: str = ""
    sevir_subject_key: str = ""
    data_policy: str = "cached"

    def __post_init__(self) -> None:
        sevir_subject_key = str(self.sevir_subject_key or "").strip().lower()
        event_id = str(self.event_id or "").strip().upper()
        upload_id = str(self.upload_id or "").strip()
        policy = str(self.data_policy or "cached").strip().lower()
        if policy not in {"cached", "live"}:
            raise ValueError("data_policy must be cached or live")
        for name, value in (
            ("sevir_subject_key", sevir_subject_key),
            ("event_id", event_id),
            ("upload_id", upload_id),
            ("data_policy", policy),
        ):
            object.__setattr__(self, name, value)

    @property
    def assets(self) -> RequestAssets:
        """Return the request's selected event and upload identities."""

        return RequestAssets(
            event_id=self.event_id,
            upload_id=self.upload_id,
            sevir_subject_key=self.sevir_subject_key,
        )

    @property
    def policy(self) -> DataAccessPolicy:
        """Return the request's data-access policy."""

        return DataAccessPolicy(self.data_policy)

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RequestBinding":
        raw = value or {}
        upload_id = str(raw.get("upload_id") or "").strip()
        sequence_upload_id = str(raw.get("sequence_upload_id") or "").strip()
        if upload_id and sequence_upload_id and upload_id != sequence_upload_id:
            raise ValueError("upload_id conflicts with sequence_upload_id")
        return cls(
            sevir_subject_key=raw.get("sevir_subject_key", ""),
            event_id=raw.get("event_id", ""),
            upload_id=sequence_upload_id or upload_id,
            data_policy=raw.get("data_policy", "cached"),
        )


class CapabilityUnavailable(RuntimeError):
    """A backend failure retaining its original machine-readable response."""

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(payload or {})
        self.code = self.details.get("code") or self.details.get("status") or "CAPABILITY_UNAVAILABLE"
        self.phase = "backend"
        self.backend_executed = True
        self.failure_class = classify_failure(self.details)
        if "retryable" in self.details:
            self.retryable = self.details["retryable"]



class BoundCapability:
    """One capability whose request identity is fixed at construction."""

    def __init__(self, recorder: Any, binding: RequestBinding) -> None:
        self._recorder = recorder
        self.request = binding
        self._cache: dict[str, Any] = {}
        self._product_cache: dict[str, dict[str, Any] | None] = {}

    def _event_id(self, explicit: str | None = None) -> str:
        candidate: Any = explicit or self.request.assets.event_id
        # Public catalog searches return CatalogEvent values. Accepting their
        # documented ``id`` attribute at the boundary prevents generated code
        # from accidentally stringifying the entire dataclass as an event ID.
        if not isinstance(candidate, (str, int, float)):
            candidate = getattr(candidate, "id", None)
            if candidate is None and isinstance(explicit, dict):
                candidate = explicit.get("id") or explicit.get("event_id")
        value = str(candidate or "").strip().upper()
        if not value:
            raise ValueError("this request has no bound SEVIR event")
        return value

    def _upload_id(self, explicit: str | None = None) -> str:
        value = str(explicit or self.request.assets.upload_id).strip()
        if not value:
            raise ValueError("this request has no bound upload")
        return value

    def _invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
        convert: Callable[[dict[str, Any]], T],
        *,
        public_arguments: dict[str, Any] | None = None,
    ) -> T:
        recorded_arguments = arguments if public_arguments is None else public_arguments
        key = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
        operation = operation_spec(name)
        invocation_identity = {
            "event_id": self.request.event_id, "upload_id": self.request.upload_id,
            "subject_key": self.request.sevir_subject_key, "arguments": deepcopy(arguments),
        } if operation.result_kind == "model_product" else None
        cache_hit = key in self._cache
        with self._recorder.work(name, recorded_arguments, cache_hit=cache_hit) as call:
            if invocation_identity is not None:
                call["invocation_identity"] = invocation_identity
            if cache_hit:
                value = self._cache[key]
                if invocation_identity is not None:
                    call["product"] = deepcopy(self._product_cache.get(key))
                return value
            raw = function(**arguments)
            call["phase"] = "result_conversion"
            if not isinstance(raw, dict):
                raise TypeError(f"{name} backend returned {type(raw).__name__}, expected dict")
            status = str(raw.get("status", "ok")).lower()
            if status not in {
                "ok", "success", "ambiguous", "approximate_boundary_match", "partial",
                "outside_us_boundaries", "boundary_data_unavailable",
                "outside_available_boundaries",
            }:
                reason = raw.get("reason") or raw.get("error") or status
                raise CapabilityUnavailable(f"{name} unavailable: {reason}", payload=raw)
            value = convert(raw)
            product = capture_product(operation, value, arguments)
            if product is not None:
                value.attrs["product_metadata"] = deepcopy(product)
                call["product"] = deepcopy(product)
            call["cache_hit"] = bool(raw.get("cache_hit", False))
        self._cache[key] = value
        self._product_cache[key] = deepcopy(product)
        return value

    def _subject_key(self, explicit: str | None = None) -> str:
        """Return the unique request-bound scientific subject identity."""

        candidate: Any = explicit or self.request.assets.sevir_subject_key
        if not candidate and self.request.assets.event_id:
            from ..tools.sevir.repository import resolve_sevir_subject_key

            candidate = resolve_sevir_subject_key(self.request.assets.event_id)
        if not isinstance(candidate, (str, int, float)):
            candidate = getattr(candidate, "subject_key", None)
            if candidate is None and isinstance(explicit, dict):
                candidate = explicit.get("subject_key")
        value = str(candidate or "").strip().lower()
        if not value:
            raise ValueError("this request has no uniquely bound scientific subject")
        return value


class ToolCostBudgetExceeded(RuntimeError):
    """Raised before a backend call would exceed the request cost budget."""


class ToolUsageRecorder:
    """Record calls and enforce a persistent weighted runtime cost budget."""

    def __init__(self, cost_budget_units: float | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        configured = (
            float(cost_budget_units)
            if cost_budget_units is not None
            else float(os.getenv("SEVIR_AGENT_TOOL_COST_BUDGET", "30"))
        )
        if configured <= 0:
            raise ValueError("tool cost budget must be positive")
        self.cost_budget_units = configured
        self.cost_used_units = 0.0
        self._active_call: ContextVar[dict[str, Any] | None] = ContextVar("tool_call", default=None)

    def _reserve(
        self, name: str, arguments: dict[str, Any], *, cache_hit: bool = False
    ) -> float:
        """Reserve cost before backend execution so loops cannot evade the budget."""

        cost = operation_spec(name).estimate_cost(arguments, cache_hit)
        projected = self.cost_used_units + cost
        if projected > self.cost_budget_units + 1e-9:
            raise ToolCostBudgetExceeded(
                f"{name} costs {cost:g} units but only "
                f"{self.remaining_cost_units:g} of {self.cost_budget_units:g} remain"
            )
        self.cost_used_units = round(projected, 2)
        return cost

    @property
    def remaining_cost_units(self) -> float:
        return round(max(0.0, self.cost_budget_units - self.cost_used_units), 2)

    def summary(self) -> dict[str, float | bool]:
        return {
            "budget_units": self.cost_budget_units,
            "used_units": self.cost_used_units,
            "remaining_units": self.remaining_cost_units,
            "exhausted": self.remaining_cost_units <= 0,
        }

    @contextmanager
    def invocation(self, name: str, arguments: dict[str, Any], *, binding: Any = None, operation=None):
        """Own one public call, including failures before any work is attempted."""
        operation = operation or operation_spec(name)
        parent = self._active_call.get()
        call: dict[str, Any] = {
            "step": len(self.calls) + 1, "call_id": uuid4().hex,
            "parent_call_id": parent["call_id"] if parent else None,
            "api": name, **operation_display(name, operation),
            "result_kind": operation.result_kind, "model_id": operation.model_id,
            "arguments": arguments, "status": "running", "phase": "method_lookup",
            "backend_executed": False, "cache_hit": False, "cost_units": 0.0,
            "work": [],
        }
        if operation.result_kind == "model_product":
            call["invocation_identity"] = {
                key: getattr(binding, key, "") for key in ("event_id", "upload_id", "sevir_subject_key")
            }
        self.calls.append(call)
        token = self._active_call.set(call)
        started = time.perf_counter()
        try:
            yield call
            call["status"] = "ok"
        except BaseException as error:
            call["status"] = "error"
            call["error"] = error_record(error, stage="tool")
            call["error"].setdefault("phase", call["phase"])
            call["error"].setdefault("backend_executed", call["backend_executed"])
            raise
        finally:
            self._active_call.reset(token)
            call["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            call["arguments"] = serialize_value(call["arguments"], summary=CALL_ARGUMENT_LIMITS)
            call["cumulative_cost_units"] = self.cost_used_units
            call["remaining_cost_units"] = self.remaining_cost_units

    @contextmanager
    def work(self, name: str, arguments: dict[str, Any], *, cache_hit: bool = False, backend: bool = True):
        """Reserve actual work once and attach facts to the active public call."""
        call = self._active_call.get()
        if call is None:
            raise RuntimeError("recorded capabilities must be called through ToolInterface")
        work = {"api": name, "cache_hit": cache_hit, "cost_units": 0.0, "backend_executed": False}
        call["work"].append(work)
        call["arguments"] = arguments
        call["phase"] = "cost_reservation"
        cost = self._reserve(name, arguments, cache_hit=cache_hit)
        work.update(cost_units=cost, backend_executed=backend and not cache_hit)
        call["cost_units"] = round(call["cost_units"] + cost, 2)
        call["backend_executed"] |= work["backend_executed"]
        call["cache_hit"] = all(item["cache_hit"] for item in call["work"])
        call["phase"] = "backend" if backend and not cache_hit else "local_execution"
        yield call


class ToolInterface:
    """The sole public invocation boundary; adapters only report work facts."""

    def __init__(
        self,
        name: str,
        component: Any,
        recorder: ToolUsageRecorder,
    ) -> None:
        self._name = str(name)
        self._component = component
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            method = getattr(self._component, name)
        except Exception:
            with self._recorder.invocation(f"{self._name}.{name}", {}):
                raise
        if not callable(method):
            return method

        def invoke(*args: Any, **kwargs: Any) -> Any:
            operation = operation_spec(f"{self._name}.{name}")
            with self._recorder.invocation(
                operation.name, {"positional": args, **kwargs},
                binding=getattr(self._component, "request", None),
                operation=operation,
            ) as call:
                call["phase"] = "input_validation"
                arguments = operation.validate(method, args, kwargs)
                call["arguments"] = arguments
                call["phase"] = "preprocessing"
                value = method(*args, **kwargs)
                call["phase"] = "result_recording"
                call["result"] = serialize_value(value, summary=CALL_RESULT_LIMITS)
                call["return_type"] = type(value).__name__
                # A generated program must not own the cache's mutable object.
                return isolated_copy(value)

        return invoke


class LazyToolInterface:
    """Load one capability only when generated code first accesses it."""

    def __init__(self, name: str, loader: Callable[[], Any], recorder: ToolUsageRecorder) -> None:
        self._name = str(name)
        self._loader = loader
        self._recorder = recorder
        self._interface: ToolInterface | None = None

    @property
    def loaded(self) -> bool:
        return self._interface is not None

    def _load(self) -> ToolInterface:
        if self._interface is None:
            self._interface = ToolInterface(self._name, self._loader(), self._recorder)
        return self._interface

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            interface = self._load()
        except Exception:
            with self._recorder.invocation(f"{self._name}.{name}", {}) as call:
                call["phase"] = "capability_loading"
                raise
        return getattr(interface, name)
