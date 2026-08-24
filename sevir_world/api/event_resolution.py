from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sevir_world.agent.session import load_session, normalize_session_id
from sevir_world.data_access.catalog import get_sevir_event_metadata
from sevir_world.data_access.request_identity import detect_event_conflict
from sevir_world.data_access.uploads import match_uploaded_event


@dataclass(frozen=True)
class EventResolutionOutcome:
    current_upload_id: str = ""
    upload_event_id: str = ""
    conflict: dict[str, Any] | None = None

    def conflict_response(self) -> dict[str, Any] | None:
        if not self.conflict:
            return None
        kind = str(self.conflict.get("kind", ""))
        if kind == "event_metadata_conflict":
            return {
                "status": "conflict",
                "answer": "问题中的时间或地点与当前上传事件不一致，已停止沿用上传事件的ERA5/Aurora证据。请确认新的UTC时间和经纬度/区域后重新请求现场分析。",
                "route_mode": "deterministic_event_conflict",
                "termination_reason": "event_metadata_conflict",
                "event_conflict": self.conflict.get("details", {}),
                "selected_tools": [],
            }
        if kind == "event_id_conflict":
            details = self.conflict.get("details", {})
            requested = str(details.get("requested", ""))
            uploaded = str(details.get("uploaded", ""))
            return {
                "status": "conflict",
                "answer": f"指定事件 {requested} 与上传数据匹配到的事件 {uploaded} 不一致，已停止混用两者证据。",
                "route_mode": "deterministic_event_conflict",
                "termination_reason": "event_id_conflict",
                "event_id_conflict": details,
                "selected_tools": [],
            }
        return None


def _resolve_current_upload(payload: dict[str, Any]) -> tuple[str, str]:
    current_upload_id = str(
        payload.get("sequence_upload_id", "") or payload.get("upload_id", "")
    ).strip()
    if not current_upload_id:
        return "", ""
    try:
        match_result = match_uploaded_event(current_upload_id)
        event_match = match_result.get("event_match", {})
        if not isinstance(event_match, dict):
            event_match = {}
        payload["event_match"] = event_match
        upload_event_id = str(event_match.get("event_id") or "").strip().upper()
        if upload_event_id:
            payload["uploaded_event_id"] = upload_event_id
            if not str(payload.get("event_id", "")).strip():
                payload["event_id"] = upload_event_id
            payload["event_resolution"] = {
                "source": "current_upload",
                "status": "matched",
                "event_id": upload_event_id,
                "method": event_match.get("method"),
                "upload_id": current_upload_id,
                "session_reuse_blocked": False,
            }
        else:
            payload["event_resolution"] = {
                "source": "current_upload",
                "status": str(event_match.get("status") or "not_found"),
                "event_id": None,
                "method": event_match.get("method"),
                "upload_id": current_upload_id,
                "session_reuse_blocked": True,
            }
        return current_upload_id, upload_event_id
    except (ValueError, OSError, KeyError) as error:
        payload["event_resolution"] = {
            "source": "current_upload",
            "status": "match_failed",
            "event_id": None,
            "upload_id": current_upload_id,
            "error_type": type(error).__name__,
            "session_reuse_blocked": True,
        }
        return current_upload_id, ""


def _inherit_session_event(
    payload: dict[str, Any], current_upload_id: str, *, is_foundation: bool
) -> None:
    if (
        str(payload.get("data_policy", "cached")).lower() == "live"
        or is_foundation
        or str(payload.get("event_id", "")).strip()
        or current_upload_id
    ):
        return
    session_state = load_session(normalize_session_id(payload.get("session_id")))
    for previous_turn in reversed(session_state.get("turns", [])):
        previous_text = " ".join(
            str(previous_turn.get(key, "")) for key in ("question", "answer")
        )
        previous_event = re.search(
            r"\bR\d{14}\b", previous_text, flags=re.IGNORECASE
        )
        if previous_event:
            event_id = previous_event.group(0).upper()
            payload["event_id"] = event_id
            payload["event_resolution"] = {
                "source": "session_history",
                "status": "inherited",
                "event_id": event_id,
                "session_reuse_blocked": False,
            }
            return


def resolve_event_context(
    question: str,
    payload: dict[str, Any],
    *,
    explicit_event_id: str = "",
    is_foundation: bool = False,
) -> EventResolutionOutcome:
    """Bind one authoritative event before final plan readiness is checked."""

    current_upload_id, upload_event_id = _resolve_current_upload(payload)
    explicit_event_id = str(explicit_event_id or "").strip().upper()
    if explicit_event_id:
        payload["event_resolution"] = {
            **(payload.get("event_resolution") or {}),
            "source": "question",
            "status": "explicit",
            "event_id": explicit_event_id,
            "upload_event_id": upload_event_id or None,
        }

    conflict_record: dict[str, Any] = {}
    conflict_event_id = upload_event_id or str(
        payload.get("event_id", "")
    ).strip().upper()
    if conflict_event_id:
        try:
            conflict_record = (
                get_sevir_event_metadata(conflict_event_id).get("record") or {}
            )
        except (OSError, ValueError, KeyError, TypeError):
            conflict_record = {}
    metadata_conflict = detect_event_conflict(question, payload, conflict_record)
    payload["event_conflict"] = metadata_conflict

    _inherit_session_event(
        payload, current_upload_id, is_foundation=bool(is_foundation)
    )

    requested_event_id = str(payload.get("event_id", "")).strip().upper()
    event_match = payload.get("event_match")
    if not isinstance(event_match, dict):
        event_match = {}
    uploaded_event_id = str(
        payload.get("uploaded_event_id", "") or event_match.get("event_id", "")
    ).strip().upper()
    conflict: dict[str, Any] | None = None
    if metadata_conflict.get("status") == "conflict" and is_foundation:
        conflict = {
            "kind": "event_metadata_conflict",
            "details": metadata_conflict,
        }
    elif (
        requested_event_id
        and uploaded_event_id
        and requested_event_id != uploaded_event_id
    ):
        conflict = {
            "kind": "event_id_conflict",
            "details": {
                "requested": requested_event_id,
                "uploaded": uploaded_event_id,
            },
        }
    return EventResolutionOutcome(current_upload_id, upload_event_id, conflict)
