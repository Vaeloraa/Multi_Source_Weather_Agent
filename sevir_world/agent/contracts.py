"""Execution data structures and pure consistency checks; no request resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from ..capabilities.base import RequestBinding


DATA_ID_FIELDS = {"upload": "upload_id", "sevir": "event_id"}
DATA_CONTRACT_VERSION = "sevir-data-v3"
# Bump this whenever the injected namespace or public capability return
# contracts change. The persistent Worker uses it as an explicit generation
# token and must not be reused across generations.
WORKER_CONTRACT_VERSION = "sevir-worker-v18"


def _never_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class ExecutionRuntime:
    """One request's runtime controls, separate from persisted request facts."""

    request_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    deadline_monotonic: float | None = None
    budget_seconds: float | None = None
    cancelled: Callable[[], bool] = field(default=_never_cancelled, compare=False)
    trace: Any = field(default=None, compare=False)
    started_perf_counter: float = field(default_factory=time.perf_counter, compare=False)

    def is_cancelled(self) -> bool:
        return bool(self.cancelled())

    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - time.monotonic()


@dataclass(frozen=True)
class ExecutionRequest:
    """One Agent execution's runtime assets and verbatim structured inputs."""

    data_descs: tuple[dict[str, Any], ...]
    selected_tools: tuple[str, ...]
    binding: RequestBinding
    explicit_inputs: dict[str, Any] = field(default_factory=dict)
    generate_visual: bool = False
    evidence_state: dict[str, Any] = field(default_factory=dict)
    # selected_tools is the immutable request allowlist. The Agent activates
    # a growing subset after reading the short catalog; it cannot widen access.
    active_tools: tuple[str, ...] | None = field(default=None, compare=False)
    # Host-only conversation sources; these do not authorize data or tools.
    conversation_context: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.generate_visual, bool):
            raise ValueError("generate_visual must be a boolean")
        if not isinstance(self.binding, RequestBinding):
            raise TypeError("binding must be a RequestBinding")
        if not isinstance(self.data_descs, tuple):
            raise TypeError("data_descs must be a tuple of descriptions")
        if not isinstance(self.selected_tools, tuple) or not all(
            isinstance(name, str) for name in self.selected_tools
        ):
            raise TypeError("selected_tools must be a tuple of capability names")
        if not isinstance(self.explicit_inputs, dict) or not isinstance(self.evidence_state, dict):
            raise TypeError("explicit_inputs and evidence_state must be objects")
        if not isinstance(self.conversation_context, dict):
            raise TypeError("conversation_context must be an object")
        if self.active_tools is not None:
            if not isinstance(self.active_tools, tuple) or not all(isinstance(name, str) for name in self.active_tools):
                raise TypeError("active_tools must be a tuple of capability names")
            if len(set(self.active_tools)) != len(self.active_tools):
                raise ValueError("active_tools contains duplicates")
            if "ds" not in self.active_tools or set(self.active_tools) - set(self.selected_tools):
                raise ValueError("active_tools must include ds and stay within the request allowlist")
        for index, item in enumerate(self.data_descs):
            if not isinstance(item, dict) or item.get("source") not in DATA_ID_FIELDS:
                raise ValueError(f"data_descs[{index}] has an invalid source")
            key = DATA_ID_FIELDS[item["source"]]
            identity = item.get(key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"data_descs[{index}] requires {key}")
            if item.get("subject_key") is not None and not isinstance(item["subject_key"], str):
                raise ValueError(f"data_descs[{index}].subject_key must be a string")
        for source, field_name, description_field in (
            ("upload", "upload_id", "upload_id"),
            ("sevir", "event_id", "event_id"),
            ("sevir", "sevir_subject_key", "subject_key"),
        ):
            records = [item for item in self.data_descs if item["source"] == source]
            if field_name == "sevir_subject_key":
                records = [item for item in records if item["event_id"] == self.binding.event_id]
            identities = {item[description_field] for item in records if item.get(description_field)}
            if identities and getattr(self.binding, field_name) not in identities:
                raise ValueError(
                    f"request identity conflict: binding.{field_name}="
                    f"{getattr(self.binding, field_name)!r} is not present in data_descs "
                    f"{description_field} values {sorted(identities)!r}"
                )
