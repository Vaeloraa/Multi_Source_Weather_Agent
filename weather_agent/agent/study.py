from __future__ import annotations

"""Structured execution evidence and explicit decisions shared across requests."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any


SCHEMA_VERSION = "study-evidence-v1"


def build_turn_evidence(
    *,
    turn_id: str,
    request_id: str,
    question: str,
    request_facts: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Record execution observations without inferring decisions from tool names."""

    observations = result.get("observations", [])
    if not observations:
        return None
    tool_evidence: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []
    derived_results: list[Any] = []
    update: dict[str, Any] = {}
    for index, observation in enumerate(observations, start=1):
        observed_result = observation.get("result")
        if observed_result is not None and observation["status"] in {"ok", "partial"}:
            derived_results.append(deepcopy(observed_result))
            if isinstance(observed_result, dict) and "study_update" in observed_result:
                update.update(deepcopy(observed_result["study_update"]))
        for call in observation.get("calls", []):
            source, _, operation = call["api"].partition(".")
            tool_evidence.append({
                **deepcopy(call),
                "observation": index,
                "source": source,
                "operation": operation,
            })
            if call.get("result_kind") == "model_product":
                product = call.get("product") or {}
                identity = {
                    "api": call["api"],
                    "model_id": product.get("model_id") or call.get("model_id"),
                    "model_version": product.get("model_version", ""),
                    "input_identity": product.get("input_identity") or call.get("invocation_identity", {}),
                    "arguments": call.get("arguments", {}),
                    "cache_identity": product.get("cache_identity", {}),
                }
                product_key = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest()[:32]
                model_runs.append({
                    "run_id": f"{turn_id}:{index}:{len(model_runs) + 1}",
                    "product_key": product_key,
                    **deepcopy(identity),
                    "status": "available" if call["status"] == "ok" and product else "failed",
                    "product": deepcopy(product),
                    "error": deepcopy(call.get("error")),
                })
    return {
        "record_id": f"{turn_id}:evidence",
        "turn_id": turn_id,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "status": result["status"],
        "request_facts": deepcopy(request_facts),
        "tool_evidence": tool_evidence,
        "model_runs": model_runs,
        "derived_results": derived_results,
        "artifacts": deepcopy(result.get("artifacts", [])),
        "study_update": update,
    }


def rebuild_study_case(
    session_id: str,
    turns: list[dict[str, Any]],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Replay explicit decisions for the requested case, or the latest case."""

    if case_id is None:
        case_id = turns[-1].get("study_case_id") if turns else None
    case_id = case_id or session_id
    records = [
        deepcopy(turn["evidence_record"])
        for turn in turns
        if turn.get("evidence_record") is not None
        and (turn.get("study_case_id") or session_id) == case_id
    ]
    decisions: dict[str, Any] = {}
    for record in records:
        for key, value in record.get("study_update", {}).items():
            if value is None:
                decisions.pop(key, None)
            else:
                decisions[key] = deepcopy(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "revision": len(records),
        "decisions": decisions,
        "model_runs": {
            run["run_id"]: run
            for record in records
            for run in record.get("model_runs", [])
        },
        "records": records,
    }


def evidence_index(state: dict[str, Any] | None) -> dict[str, Any]:
    """Project an index for clients; full evidence stays in the ledger."""

    value = state or {}
    records = []
    for record in value.get("records", []):
        calls = record.get("tool_evidence", [])
        records.append({
            "record_id": record["record_id"],
            "turn_id": record["turn_id"],
            "question": record.get("question", ""),
            "status": record.get("status"),
            "sources": sorted({call["source"] for call in calls}),
            "operations": sorted({f"{call['source']}.{call['operation']}" for call in calls}),
            "request_facts": record.get("request_facts", {}),
            "study_update": record.get("study_update", {}),
            "artifact_count": len(record.get("artifacts", [])),
        })
    return deepcopy({
        "schema_version": value.get("schema_version", SCHEMA_VERSION),
        "case_id": value.get("case_id", ""),
        "revision": value.get("revision", 0),
        "decisions": value.get("decisions", {}),
        "model_runs": {
            key: {name: run.get(name) for name in (
                "run_id", "product_key", "api", "model_id", "model_version", "status",
            )}
            for key, run in value.get("model_runs", {}).items()
        },
        "records": records,
    })


class StudyEvidence:
    """Read-only access to the complete request-bound evidence ledger."""

    def __init__(self, state: dict[str, Any] | None) -> None:
        self._state = deepcopy(state or {})

    @property
    def case_id(self) -> str:
        return self._state.get("case_id", "")

    @property
    def revision(self) -> int:
        return self._state.get("revision", 0)

    @property
    def model_runs(self) -> dict[str, Any]:
        """Return recorded model invocations, products and failures."""
        return deepcopy(self._state.get("model_runs", {}))

    @property
    def decisions(self) -> dict[str, Any]:
        """Return decisions explicitly published by prior executions."""
        return deepcopy(self._state.get("decisions", {}))

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return evidence records in chronological order."""
        return tuple(deepcopy(self._state.get("records", [])))

    def __iter__(self):
        return iter(self.records())

    def select(
        self,
        *,
        source: str | None = None,
        operation: str | None = None,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Match source and operation on the same recorded tool call."""
        selected = []
        for record in self._state.get("records", []):
            if turn_id is not None and record["turn_id"] != turn_id:
                continue
            if (source is not None or operation is not None) and not any(
                (source is None or call["source"] == source)
                and (operation is None or call["operation"] == operation)
                for call in record.get("tool_evidence", [])
            ):
                continue
            selected.append(deepcopy(record))
        return tuple(selected)

    def latest(
        self,
        *,
        source: str | None = None,
        operation: str | None = None,
    ) -> dict[str, Any] | None:
        matches = self.select(source=source, operation=operation)
        return matches[-1] if matches else None

    def get(self, record_id: str) -> dict[str, Any]:
        for record in self._state.get("records", []):
            if record["record_id"] == record_id:
                return deepcopy(record)
        raise KeyError(f"unknown study evidence record: {record_id}")
