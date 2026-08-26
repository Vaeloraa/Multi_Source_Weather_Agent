from __future__ import annotations

"""Durable conversation history for the unified Agent."""

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data_access.config import ROOT
from .study import prompt_context, rebuild_study_case


_DEFAULT_ROOT = ROOT / "data" / "runtime" / "agent_sessions"
_FALLBACK_ROOT = Path(tempfile.gettempdir()) / "sevir_world_runtime" / "agent_sessions"
_LOCK = threading.RLock()
_MEMORY: dict[str, dict[str, Any]] = {}


def _session_root() -> Path:
    candidate = Path(os.getenv("SEVIR_AGENT_SESSION_DIR", str(_DEFAULT_ROOT)))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        _FALLBACK_ROOT.mkdir(parents=True, exist_ok=True)
        return _FALLBACK_ROOT


SESSION_ROOT = _session_root()
LATEST_SESSION_PATH = SESSION_ROOT / "_latest_session.json"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _bounded(value: Any, depth: int = 0) -> Any:
    """Keep conversational provenance compact without deriving new answers."""
    if depth > 4:
        return {"type": type(value).__name__, "omitted": True}
    if isinstance(value, dict):
        return {
            str(key): _bounded(item, depth + 1)
            for key, item in list(value.items())[:30]
            if key not in {"traceback", "code", "raw_data", "values", "frames"}
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > 12:
            items = [*items[:4], {"omitted_items": len(items) - 5}, items[-1]]
        return [_bounded(item, depth + 1) for item in items]
    if isinstance(value, str):
        return value[:2000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def normalize_session_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return uuid.uuid4().hex
    if len(raw) > 80 or not all(char.isalnum() or char in "-_" for char in raw):
        raise ValueError("session_id只能包含字母、数字、连字符和下划线")
    return raw


def normalize_turn_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return uuid.uuid4().hex
    if len(raw) > 80 or not all(char.isalnum() or char in "-_" for char in raw):
        raise ValueError("turn_id只能包含字母、数字、连字符和下划线")
    return raw


def _path(session_id: str) -> Path:
    return SESSION_ROOT / f"{session_id}.json"


def load_session(session_id: str) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    with _LOCK:
        if session_id in _MEMORY:
            return _json_copy(_MEMORY[session_id])
        path = _path(session_id)
        if not path.is_file():
            return {"session_id": session_id, "turns": []}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"session_id": session_id, "turns": []}
        if not isinstance(value, dict) or not isinstance(value.get("turns"), list):
            return {"session_id": session_id, "turns": []}
        value["session_id"] = session_id
        value["turns"] = [item for item in value["turns"] if isinstance(item, dict)]
        value["study_case"] = rebuild_study_case(
            session_id,
            value["turns"],
            value.get("study_case") if isinstance(value.get("study_case"), dict) else None,
        )
        return value


def latest_session_id() -> str:
    try:
        value = json.loads(LATEST_SESSION_PATH.read_text(encoding="utf-8"))
        candidate = str(value.get("session_id", ""))
        if candidate and load_session(candidate).get("turns"):
            return candidate
    except (OSError, ValueError, TypeError):
        pass
    sessions = available_session_ids()
    return sessions[0] if sessions else ""


def available_session_ids() -> list[str]:
    try:
        paths = [
            path for path in SESSION_ROOT.glob("*.json")
            if path.name != LATEST_SESSION_PATH.name
        ]
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return [
            path.stem for path in paths
            if load_session(path.stem).get("turns")
        ]
    except OSError:
        return [
            key for key, value in reversed(list(_MEMORY.items()))
            if value.get("turns")
        ]


def _save(session_id: str, state: dict[str, Any]) -> None:
    try:
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = _path(session_id).with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8"
        )
        temporary.replace(_path(session_id))
        LATEST_SESSION_PATH.write_text(
            json.dumps({"session_id": session_id}), encoding="utf-8"
        )
        _MEMORY.pop(session_id, None)
    except OSError:
        _MEMORY[session_id] = _json_copy(state)


def append_turn(session_id: str, turn: dict[str, Any]) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    with _LOCK:
        state = load_session(session_id)
        record = {
            "turn_id": normalize_turn_id(turn.get("turn_id")),
            "created_at": str(
                turn.get("created_at") or datetime.now(timezone.utc).isoformat()
            ),
            "question": str(turn.get("question", ""))[:8000],
            "answer": str(turn.get("answer", ""))[:12000],
            "selected_tools": list(turn.get("selected_tools", [])),
            "termination_reason": str(turn.get("termination_reason", "unknown")),
            "code_execution_count": int(turn.get("code_execution_count", 0) or 0),
            "evidence_execution_count": int(turn.get("evidence_execution_count", 0) or 0),
            "evidence_tool_call_count": int(turn.get("evidence_tool_call_count", 0) or 0),
            "postprocessing_success_count": int(turn.get("postprocessing_success_count", 0) or 0),
            "tool_cost_summary": _bounded(turn.get("tool_cost_summary", {})),
            "artifacts": _bounded(turn.get("artifacts", [])),
            "observations": _bounded(turn.get("observations", [])),
            "evidence_record": (
                _json_copy(turn.get("evidence_record"))
                if isinstance(turn.get("evidence_record"), dict)
                else None
            ),
            "study_case_id": str(
                turn.get("study_case_id")
                or (
                    state.get("study_case", {}).get("case_id")
                    if isinstance(state.get("study_case"), dict)
                    else ""
                )
                or session_id
            ),
        }
        turns = list(state.get("turns", []))
        match = next(
            (index for index, item in enumerate(turns)
             if item.get("turn_id") == record["turn_id"]),
            None,
        )
        if match is None:
            turns.append(record)
        else:
            turns[match] = record
        limit = max(0, int(os.getenv("SEVIR_SESSION_TURN_LIMIT", "0")))
        state["turns"] = turns if not limit else turns[-limit:]
        requested_case_id = str(turn.get("study_case_id") or "").strip()
        previous_case = state.get("study_case") if isinstance(state.get("study_case"), dict) else {}
        if requested_case_id:
            if requested_case_id == str(previous_case.get("case_id") or session_id):
                previous_case = {**previous_case, "case_id": requested_case_id}
            else:
                previous_case = {"case_id": requested_case_id}
        state["study_case"] = rebuild_study_case(
            session_id, state["turns"], previous_case
        )
        upload = turn.get("active_upload")
        if isinstance(upload, dict) and (
            upload.get("upload_id") or upload.get("sequence_upload_id")
        ):
            state["active_upload"] = _bounded(upload)
        _save(session_id, state)
        return _json_copy(state)


def delete_turn(
    session_id: str,
    turn_id: str,
    question: str = "",
    answer: str = "",
) -> dict[str, Any]:
    session_id = normalize_session_id(session_id)
    turn_id = normalize_turn_id(turn_id)
    with _LOCK:
        state = load_session(session_id)
        before = list(state.get("turns", []))
        after = [item for item in before if str(item.get("turn_id")) != turn_id]
        if len(after) == len(before) and question:
            removed = False
            kept = []
            for item in reversed(before):
                matches = (
                    not removed
                    and str(item.get("question", "")) == str(question)
                    and (not answer or str(item.get("answer", "")) == str(answer))
                )
                if matches:
                    removed = True
                else:
                    kept.append(item)
            after = list(reversed(kept))
        state["turns"] = after
        state["study_case"] = rebuild_study_case(
            session_id,
            after,
            state.get("study_case") if isinstance(state.get("study_case"), dict) else None,
        )
        _save(session_id, state)
        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "deleted": len(after) != len(before),
            "session_turn_count": len(after),
        }


def context_for_prompt(state: dict[str, Any]) -> str:
    """Return only bounded structured evidence for a continuing Study Case.

    Re-sending recent questions and answer prose beside the evidence ledger
    duplicated the user's goal on every turn and encouraged the Agent to
    restate earlier conclusions.  Prior narrative remains persisted for UI
    history, but is deliberately not part of the analytical prompt.
    """

    return prompt_context(state.get("study_case"))
