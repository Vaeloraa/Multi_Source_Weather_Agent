"""Prepare durable Agent task submissions and retries."""
from __future__ import annotations

import hashlib
import time
import uuid

from ..agent.session import normalize_session_id, normalize_turn_id
from ..agent.service import request_budget_seconds


TASK_REQUEST_ENVELOPE_SCHEMA_VERSION = "task-request-envelope-v2"


def sanitized_replay_payload(value):
    sensitive = {
        "authorization", "api_key", "apikey", "access_token", "refresh_token",
        "password", "secret", "credential", "credentials",
    }
    if isinstance(value, dict):
        return {str(key): sanitized_replay_payload(item) for key, item in value.items()
                if str(key).strip().lower() not in sensitive}
    if isinstance(value, list):
        return [sanitized_replay_payload(item) for item in value]
    return value


def submission_arguments(payload: dict, header_key: str = "") -> dict:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    generate_visual = payload.get("generate_visual", False)
    if not isinstance(generate_visual, bool):
        raise ValueError("generate_visual must be a boolean")
    key = str(payload.get("idempotency_key") or header_key or "").strip()
    request_id = str(payload.get("request_id") or "").strip() or (
        "task-" + hashlib.sha256(key.encode()).hexdigest()[:32] if key else uuid.uuid4().hex
    )
    stable_id = hashlib.sha256(request_id.encode()).hexdigest()[:24]
    session_id = normalize_session_id(payload.get("session_id") or f"task-session-{stable_id}")
    turn_id = normalize_turn_id(payload.get("turn_id") or f"task-turn-{stable_id}")
    submitted_at = time.time()
    budget = request_budget_seconds(
        payload.get("data_policy"), override_seconds=payload.get("_request_budget_seconds"),
    )
    external = sanitized_replay_payload(dict(payload))
    external.update(question=question, generate_visual=generate_visual)
    return {
        "request_id": request_id, "idempotency_key": key or f"task-submit:{request_id}",
        "session_id": session_id, "turn_id": turn_id,
        "request": {
            "schema_version": TASK_REQUEST_ENVELOPE_SCHEMA_VERSION,
            "deadline_at_epoch": submitted_at + budget, "budget_seconds": budget,
            "payload": external,
        },
    }


def retry_arguments(parent_id: str, payload: dict, header_key: str = "") -> dict:
    key = str(payload.get("idempotency_key") or header_key or "").strip()
    request_id = str(payload.get("request_id") or "").strip() or (
        "retry-" + hashlib.sha256(f"{parent_id}:{key}".encode()).hexdigest()[:32]
        if key else uuid.uuid4().hex
    )
    return {
        "request_id": request_id, "idempotency_key": key or f"task-retry:{parent_id}:{request_id}",
        "turn_id": normalize_turn_id(payload["turn_id"]) if payload.get("turn_id") is not None else None,
    }
