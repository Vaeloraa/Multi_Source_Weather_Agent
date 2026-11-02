from __future__ import annotations

"""Persistent conversation history, independent of evidence projections."""

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import ROOT
from ..persistence.sqlite_sessions import SQLiteSessionStore
from .case_memory import CaseMemoryStore, case_memory_path
from .study import build_turn_evidence, evidence_index, rebuild_study_case


SESSION_ROOT = Path(os.getenv(
    "SEVIR_AGENT_SESSION_DIR", str(ROOT / "data" / "runtime" / "agent_sessions")
))


class TurnIdentityConflict(RuntimeError):
    """Raised when one turn ID is reused by a different request."""


def normalize_session_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return uuid.uuid4().hex
    if len(raw) > 80 or not all(char.isalnum() or char in "-_" for char in raw):
        raise ValueError("session_id只能包含字母、数字、连字符和下划线")
    return raw


def normalize_turn_id(value: Any) -> str:
    return normalize_session_id(value)


def session_database_path(session_root: str | Path | None = None) -> Path:
    return Path(session_root or SESSION_ROOT) / "_sessions.sqlite3"


def _backend(connection: sqlite3.Connection | None = None):
    return SQLiteSessionStore(session_database_path(), connection=connection)


def load_session(session_id: str) -> dict[str, Any]:
    return _backend().load(normalize_session_id(session_id))


def latest_session_id() -> str:
    sessions = available_session_ids()
    return sessions[0] if sessions else ""


def available_session_ids() -> list[str]:
    return _backend().available_ids()


def append_turn(session_id: str, turn: dict[str, Any], *, connection: sqlite3.Connection | None = None) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    record = {
        **turn,
        "turn_id": normalize_turn_id(turn.get("turn_id")),
        "created_at": str(turn.get("created_at") or datetime.now(timezone.utc).isoformat()),
    }
    backend = _backend(connection)
    with backend.transaction(session_id):
        existing = backend.get_turn(session_id, record["turn_id"])
        existing_request_id = str((existing or {}).get("request_id") or "")
        if existing_request_id and existing_request_id != record.get("request_id", ""):
            raise TurnIdentityConflict(
                f"turn {record['turn_id']!r} belongs to request "
                f"{existing_request_id!r}, not {record.get('request_id', '')!r}"
            )
        backend.save_turn(session_id, record)
        state = backend.load(session_id)
    state["_persistence_status"] = "durable"
    return state


def delete_turn(session_id: str, turn_id: str) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    turn_id = normalize_turn_id(turn_id)
    backend = _backend()
    with backend.transaction(session_id):
        deleted = backend.delete_turn(session_id, turn_id)
        count = backend.turn_count(session_id)
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "deleted": bool(deleted),
        "session_turn_count": count,
    }


def update_case_index(session_id: str, record: dict[str, Any]) -> None:
    """The searchable index is rebuildable; source records remain authoritative."""
    try:
        CaseMemoryStore(case_memory_path(SESSION_ROOT)).upsert_turn(session_id, record)
    except (OSError, sqlite3.Error):
        logging.getLogger(__name__).exception("Case index update failed; rebuild the index")


TURN_COMMIT_SCHEMA_VERSION = "agent-turn-commit-v1"

_REQUEST_FACT_FIELDS = (
    "event_id", "sevir_subject_key", "upload_id", "sequence_upload_id",
    "place_name", "requested_time_utc", "aurora_initialization_time_utc",
    "latitude", "longitude", "region", "aurora_region", "forecast_leads_hours",
    "history_before_hours", "history_after_hours", "weather_model", "data_policy",
)


def append_session_turn(session_id: str, turn: dict[str, Any]) -> dict[str, Any]:
    """Save the source record before updating its derived case-memory index."""
    session_id = normalize_session_id(session_id)
    record = {**turn, "turn_id": normalize_turn_id(turn.get("turn_id"))}
    saved = append_turn(session_id, record)
    update_case_index(session_id, record)
    return saved


def delete_session_turn(session_id: str, turn_id: str) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    turn_id = normalize_turn_id(turn_id)
    result = delete_turn(session_id, turn_id)
    try:
        CaseMemoryStore(case_memory_path(SESSION_ROOT)).delete_turn(session_id, turn_id)
    except (OSError, sqlite3.Error):
        logging.getLogger(__name__).exception("Case index deletion failed; rebuild the index")
    return result


def merge_sessions(source_ids: list[str], target: str | None) -> dict[str, Any]:
    source_ids = [normalize_session_id(item) for item in source_ids]
    target = normalize_session_id(target)
    if len(source_ids) < 2 or target in source_ids:
        raise ValueError("合并至少需要两个不同的源对话。")
    turns = [
        (source, turn)
        for source in source_ids
        for turn in load_session(source).get("turns", [])
    ]
    turns.sort(key=lambda item: str(item[1].get("created_at", "")))
    state = {"turns": []}
    for source, turn in turns:
        turn_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{target}:{source}:{turn['turn_id']}").hex
        copied = {**turn, "turn_id": turn_id}
        if isinstance(turn.get("evidence_record"), dict):
            copied["evidence_record"] = {
                **turn["evidence_record"],
                "turn_id": turn_id,
                "record_id": f"{turn_id}:evidence",
            }
        state = append_session_turn(target, copied)
    return {
        "status": "ok",
        "session_id": target,
        "turn_count": len(state.get("turns", [])),
    }


def build_turn_commit(
    result: dict[str, Any], payload: dict[str, Any], question: str
) -> dict[str, Any] | None:
    """Build the conversation turn saved alongside the task result."""

    session_id = result.get("session_id") or payload.get("session_id")
    if not session_id:
        return None
    session_id = normalize_session_id(session_id)
    turn_id = normalize_turn_id(
        result.get("session_turn_id") or payload.get("turn_id")
    )
    request_id = str(
        result.get("request_id") or payload.get("request_id") or ""
    )
    evidence_record = build_turn_evidence(
        turn_id=turn_id,
        request_id=request_id,
        question=question,
        request_facts={key: payload[key] for key in _REQUEST_FACT_FIELDS if key in payload},
        result=result,
    )
    record = {
        "turn_id": turn_id,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": str(result.get("answer", "")),
        "status": str(result.get("status") or "unknown"),
        "presentation": dict(result.get("presentation") or {}),
        "selected_tools": list(result.get("selected_tools", [])),
        "termination_reason": str(result.get("termination_reason", "unknown")),
        "code_execution_count": int(result.get("code_execution_count", 0) or 0),
        "evidence_execution_count": int(result.get("evidence_execution_count", 0) or 0),
        "evidence_tool_call_count": int(result.get("evidence_tool_call_count", 0) or 0),
        "postprocessing_success_count": int(result.get("postprocessing_success_count", 0) or 0),
        "tool_cost_summary": dict(result.get("tool_cost_summary") or {}),
        "artifacts": list(result.get("artifacts", [])),
        "observations": list(result.get("observations", [])),
        "evidence_record": evidence_record,
        "context_summary": dict(result.get("context_summary") or {}),
        "context_archive": dict(result.get("context_archive") or {}),
        "study_case_id": str(payload.get("study_case_id") or session_id),
        "active_upload": {
            "upload_id": str(payload.get("upload_id", "")),
            "sequence_upload_id": str(payload.get("sequence_upload_id", "")),
            "data_source": str(payload.get("data_source", "upload")),
        },
    }
    result["session_id"] = session_id
    result["session_turn_id"] = turn_id
    return {
        "schema_version": TURN_COMMIT_SCHEMA_VERSION,
        "session_id": session_id,
        "turn_id": turn_id,
        "request_id": request_id,
        "record": record,
    }


def commit_turn_commit(
    turn_commit: Mapping[str, Any], *, connection: sqlite3.Connection | None = None
) -> dict[str, Any]:
    """Idempotently apply a frozen turn projection and return its receipt."""

    if str(turn_commit.get("schema_version") or "") != TURN_COMMIT_SCHEMA_VERSION:
        raise ValueError("unsupported turn commit schema")
    session_id = normalize_session_id(turn_commit.get("session_id"))
    turn_id = normalize_turn_id(turn_commit.get("turn_id"))
    record = turn_commit.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("turn commit record is missing")
    record_value = dict(record)
    if normalize_turn_id(record_value.get("turn_id")) != turn_id:
        raise ValueError("turn commit identity does not match its record")
    request_id = str(turn_commit.get("request_id") or "")
    if str(record_value.get("request_id") or "") != request_id:
        raise ValueError("turn commit request identity does not match its record")
    saved = append_turn(session_id, record_value, connection=connection)
    if connection is None:
        update_case_index(session_id, record_value)
    persistence_status = str(saved.pop("_persistence_status", "durable"))
    result_updates = {
        "session_id": session_id,
        "session_turn_id": turn_id,
        "session_turn_count": len(saved.get("turns", [])),
        "study_case": evidence_index(rebuild_study_case(session_id, saved.get("turns", []))),
        "turn_persistence": {
            "status": persistence_status,
            "schema_version": TURN_COMMIT_SCHEMA_VERSION,
        },
    }
    return {
        "schema_version": TURN_COMMIT_SCHEMA_VERSION,
        "session_id": session_id,
        "turn_id": turn_id,
        "request_id": request_id,
        "session_turn_count": result_updates["session_turn_count"],
        "study_case": result_updates["study_case"],
        "result_updates": result_updates,
    }
