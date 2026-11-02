from __future__ import annotations

"""Single-host SQLite task queue; results and conversation turns commit together."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator, Mapping

from ..config import ROOT
from ..agent.session import session_database_path
from .sqlite_sessions import SQLiteSessionStore


TASK_STORE_SCHEMA_VERSION = "task-store-v3"
_FINGERPRINT_SCHEMA_VERSION = "task-store-v1"

DEFAULT_TASK_STORE_PATH = session_database_path()
LEGACY_TASK_STORE_PATH = ROOT / "data" / "runtime" / "task_store.sqlite3"

TASK_STATES = frozenset({
    "queued",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
})
TERMINAL_TASK_STATES = frozenset({
    "succeeded", "failed", "cancelled", "interrupted",
})
RETRYABLE_TASK_STATES = frozenset({"failed", "cancelled", "interrupted"})
PERSISTENCE_STATUSES = frozenset({"durable", "degraded", "failed"})

LEGAL_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled", "failed"}),
    "running": frozenset({
        "cancelling", "succeeded", "failed", "cancelled", "interrupted",
    }),
    "cancelling": frozenset({"cancelled", "failed", "interrupted"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}

_REQUEST_ID = re.compile(r"^[0-9A-Za-z_-]{8,128}$")


class TaskStoreError(RuntimeError):
    """Base class for durable task-store contract failures."""


class TaskNotFound(TaskStoreError):
    """Raised when a request identifier is absent."""


class TaskSchemaMismatch(TaskStoreError):
    """Raised when the database has an unsupported schema version."""


class IllegalTaskTransition(TaskStoreError):
    """Raised when a task-state edge is not part of the formal state machine."""


class IdempotencyConflict(TaskStoreError):
    """Raised when one idempotency key is reused for different task input."""


class RetryNotAllowed(TaskStoreError):
    """Raised when retry lineage is requested from a non-retryable task."""


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, set):
        cleaned = [_clean_json(item) for item in value]
        return sorted(
            cleaned,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        _clean_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise TaskStoreError("invalid persisted task JSON") from error


def _task_fingerprint(
    *,
    session_id: str,
    turn_id: str,
    request: Mapping[str, Any],
    parent_request_id: str | None,
    attempt_number: int,
) -> str:
    request_identity = _clean_json(request)
    if (
        isinstance(request_identity, dict)
        and str(request_identity.get("schema_version") or "").startswith(
            "task-request-envelope-v"
        )
    ):
        # Queue timing and a geocoder's prepared snapshot are replay metadata,
        # not caller input. Excluding them makes identical idempotent submits
        # stable while the original external payload remains part of the
        # fingerprint.
        request_identity = dict(request_identity)
        for field in (
            "submitted_at_epoch", "deadline_at_epoch", "prepared_payload",
        ):
            request_identity.pop(field, None)
    canonical = {
        # Keep the fingerprint contract stable across storage-only migrations.
        "schema_version": _FINGERPRINT_SCHEMA_VERSION,
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "request": request_identity,
        "parent_request_id": str(parent_request_id or ""),
        "attempt_number": int(attempt_number),
    }
    return hashlib.sha256(_json_text(canonical).encode("utf-8")).hexdigest()


def _validate_request_id(value: str, *, field: str = "request_id") -> str:
    normalized = str(value or "").strip()
    if not _REQUEST_ID.fullmatch(normalized):
        raise ValueError(
            f"{field} must contain 8-128 letters, digits, underscores, or hyphens"
        )
    return normalized


def _required_text(value: str, *, field: str, maximum: int = 256) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{field} must contain 1-{maximum} characters")
    return normalized


class TaskStore:
    """One SQLite transaction owns each queue change and its saved answer."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else session_database_path()
        SQLiteSessionStore(self.path).available_ids()
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            self._initialize(conn)
        if path is None:
            self._import_legacy_tasks(LEGACY_TASK_STORE_PATH)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10.0, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _initialize(conn: sqlite3.Connection) -> None:
        state_values = ",".join(f"'{s}'" for s in sorted(TASK_STATES))
        persistence_values = ",".join(f"'{s}'" for s in sorted(PERSISTENCE_STATUSES))
        conn.execute("CREATE TABLE IF NOT EXISTS task_store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        version = conn.execute("SELECT value FROM task_store_meta WHERE key='schema_version'").fetchone()
        if version and version[0] not in {"task-store-v2", TASK_STORE_SCHEMA_VERSION}:
            raise TaskSchemaMismatch(f"unsupported task schema: {version[0]}")
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS tasks (
                request_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                input_fingerprint TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ({state_values})),
                revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
                request_json TEXT NOT NULL,
                result_ref TEXT,
                result_json TEXT NOT NULL DEFAULT '{{}}',
                cancel_requested INTEGER NOT NULL DEFAULT 0
                    CHECK(cancel_requested IN (0, 1)),
                parent_request_id TEXT REFERENCES tasks(request_id),
                attempt_number INTEGER NOT NULL DEFAULT 1
                    CHECK(attempt_number >= 1),
                owner_instance_id TEXT,
                heartbeat_at REAL,
                persistence_status TEXT NOT NULL
                    CHECK(persistence_status IN ({persistence_values})),
                failure_class TEXT,
                termination_reason TEXT,
                error_json TEXT NOT NULL DEFAULT '{{}}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_session_state
                ON tasks(session_id, state);
            CREATE INDEX IF NOT EXISTS idx_tasks_parent
                ON tasks(parent_request_id, attempt_number);
            CREATE INDEX IF NOT EXISTS idx_tasks_state_updated
                ON tasks(state, updated_at);
            CREATE TABLE IF NOT EXISTS task_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL REFERENCES tasks(request_id)
                    ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK(sequence >= 1),
                event_type TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                payload_json TEXT NOT NULL DEFAULT '{{}}',
                created_at REAL NOT NULL,
                UNIQUE(request_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_task_events_request
                ON task_events(request_id, sequence);
            """)
        conn.execute("INSERT OR REPLACE INTO task_store_meta VALUES ('schema_version', ?)", (TASK_STORE_SCHEMA_VERSION,))

    def _import_legacy_tasks(self, source: Path) -> None:
        """Copy the former task database once, leaving its source untouched."""
        if not source.exists() or source.resolve() == self.path.resolve():
            return
        with self._connection() as conn:
            conn.execute("ATTACH DATABASE ? AS legacy", (source.resolve().as_uri() + "?mode=ro",))
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM task_store_meta WHERE key='legacy_tasks_imported'").fetchone():
                return
            pending = conn.execute("""SELECT COUNT(*) FROM legacy.task_outcome_stages s
                JOIN legacy.tasks t USING(request_id)
                WHERE t.state NOT IN ('succeeded','failed','cancelled','interrupted')""").fetchone()[0]
            if pending:
                raise TaskStoreError("finish pending results with the previous service before importing task history")
            conn.execute("INSERT OR IGNORE INTO tasks SELECT * FROM legacy.tasks")
            conn.execute("""INSERT INTO task_events
                (request_id, sequence, event_type, from_state, to_state, revision, payload_json, created_at)
                SELECT request_id, sequence, event_type, from_state, to_state, revision, payload_json, created_at
                FROM legacy.task_events WHERE NOT EXISTS (SELECT 1 FROM task_events e
                    WHERE e.request_id=legacy.task_events.request_id AND e.sequence=legacy.task_events.sequence)""")
            conn.execute("INSERT INTO task_store_meta VALUES ('legacy_tasks_imported', '1')")

    @staticmethod
    def _task_row(row) -> dict[str, Any]:
        task = dict(row)
        for field in ("request_json", "result_json", "error_json"):
            task[field.removesuffix("_json")] = _json_value(task.pop(field))
        task["cancel_requested"] = bool(task["cancel_requested"])
        task["terminal"] = task["state"] in TERMINAL_TASK_STATES
        return task

    @staticmethod
    def _get(conn, request_id):
        row = conn.execute("SELECT * FROM tasks WHERE request_id=?", (_validate_request_id(request_id),)).fetchone()
        if row is None:
            raise TaskNotFound(request_id)
        return row

    @staticmethod
    def _event(conn, row, event_type, from_state=None, payload=None):
        sequence = conn.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM task_events WHERE request_id=?",
                                (row["request_id"],)).fetchone()[0]
        conn.execute("""INSERT INTO task_events
            (request_id,sequence,event_type,from_state,to_state,revision,payload_json,created_at)
            VALUES (?,?,?,?,?,?,?,?)""", (row["request_id"], sequence, event_type, from_state,
                row["state"], row["revision"], _json_text(payload or {}), row["updated_at"]))

    def _create(self, conn, *, request_id, idempotency_key, session_id, turn_id, request,
                parent_request_id=None, attempt_number=1, now=None):
        request_id = _validate_request_id(request_id)
        idempotency_key = _required_text(idempotency_key, field="idempotency_key")
        session_id = _required_text(session_id, field="session_id", maximum=80)
        turn_id = _required_text(turn_id, field="turn_id", maximum=80)
        fingerprint = _task_fingerprint(session_id=session_id, turn_id=turn_id, request=request,
                                       parent_request_id=parent_request_id, attempt_number=attempt_number)
        rows = conn.execute("SELECT * FROM tasks WHERE request_id=? OR idempotency_key=?",
                            (request_id, idempotency_key)).fetchall()
        if rows:
            if len(rows) == 1 and rows[0]["idempotency_key"] == idempotency_key and rows[0]["input_fingerprint"] == fingerprint:
                return self._task_row(rows[0]), False
            raise IdempotencyConflict(f"conflicting submission: {request_id}")
        timestamp = time.time() if now is None else float(now)
        conn.execute("""INSERT INTO tasks
            (request_id,idempotency_key,input_fingerprint,session_id,turn_id,state,request_json,
             parent_request_id,attempt_number,persistence_status,created_at,updated_at)
            VALUES (?,?,?,?,?,'queued',?,?,?,'durable',?,?)""",
            (request_id,idempotency_key,fingerprint,session_id,turn_id,_json_text(request),
             parent_request_id,attempt_number,timestamp,timestamp))
        row = self._get(conn, request_id)
        self._event(conn, row, "task_created")
        return self._task_row(row), True

    def create_task(self, *, request_id, idempotency_key, session_id, turn_id, request, now=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._create(conn, request_id=request_id, idempotency_key=idempotency_key,
                                session_id=session_id, turn_id=turn_id, request=request, now=now)

    def get_task(self, request_id):
        with self._connection() as conn:
            return self._task_row(self._get(conn, request_id))

    def list_tasks(self, *, session_id=None, states=None):
        clauses, values = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            values.append(session_id)
        if states is not None:
            if set(states) - TASK_STATES:
                raise ValueError("invalid task states")
            if not states:
                return []
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            values.extend(sorted(states))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as conn:
            return [self._task_row(r) for r in conn.execute(
                "SELECT * FROM tasks" + where + " ORDER BY created_at,request_id", values)]

    def list_events(self, request_id, *, after_sequence=0):
        with self._connection() as conn:
            self._get(conn, request_id)
            rows = conn.execute("SELECT * FROM task_events WHERE request_id=? AND sequence>? ORDER BY sequence",
                                (request_id, max(0, int(after_sequence)))).fetchall()
        events = [dict(r) for r in rows]
        for event in events:
            event["payload"] = _json_value(event.pop("payload_json"))
        return events

    def claim_next(self, owner_instance_id, *, now=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""SELECT * FROM tasks WHERE state='queued' AND cancel_requested=0
                AND NOT EXISTS (SELECT 1 FROM tasks active WHERE active.session_id=tasks.session_id
                    AND active.state IN ('running','cancelling'))
                ORDER BY created_at,request_id LIMIT 1""").fetchone()
            if row is None:
                return None
            timestamp = time.time() if now is None else float(now)
            conn.execute("""UPDATE tasks SET state='running', revision=revision+1, owner_instance_id=?,
                started_at=?, updated_at=? WHERE request_id=?""",
                (owner_instance_id, timestamp, timestamp, row["request_id"]))
            claimed = self._get(conn, row["request_id"])
            self._event(conn, claimed, "task_claimed", "queued")
            return self._task_row(claimed)

    def request_cancel(self, request_id, *, requested_by="user", now=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get(conn, request_id)
            if row["state"] in TERMINAL_TASK_STATES or row["cancel_requested"]:
                return self._task_row(row)
            state = "cancelled" if row["state"] == "queued" else "cancelling"
            timestamp = time.time() if now is None else float(now)
            conn.execute("""UPDATE tasks SET state=?,cancel_requested=1,revision=revision+1,
                updated_at=?,finished_at=?,termination_reason=? WHERE request_id=?""",
                (state,timestamp,timestamp if state == "cancelled" else None,
                 "cancelled_before_execution" if state == "cancelled" else None,request_id))
            updated = self._get(conn, request_id)
            self._event(conn, updated, "cancel_requested", row["state"], {"requested_by": requested_by})
            return self._task_row(updated)

    def complete_task(self, request_id, *, final_state, result=None, turn_commit=None,
                      owner_instance_id=None, result_ref=None, failure_class=None,
                      termination_reason=None, error=None):
        """Save the session turn and terminal task result in one transaction."""
        from ..agent.session import commit_turn_commit, update_case_index

        if final_state not in TERMINAL_TASK_STATES:
            raise ValueError("final_state must be terminal")
        index_turn = None
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get(conn, request_id)
            if row["state"] in TERMINAL_TASK_STATES:
                return self._task_row(row)
            if owner_instance_id and row["owner_instance_id"] != owner_instance_id:
                raise TaskStoreError("task belongs to another runner")
            resolved = "cancelled" if row["cancel_requested"] else final_state
            if resolved not in LEGAL_TRANSITIONS[row["state"]]:
                raise IllegalTaskTransition(f"cannot complete {row['state']} as {resolved}")
            result = dict(result or {})
            if resolved == "cancelled":
                result.update(status="cancelled", task_state="cancelled")
                termination_reason = "cancel_requested"
                # Cancelled executions cannot publish decisions or an old successful answer.
                result["answer"] = "本轮分析已停止。"
                turn_commit = None
            if turn_commit is not None:
                if any(str(turn_commit.get(key) or "") != str(row[key])
                       for key in ("session_id", "turn_id", "request_id")):
                    raise TaskStoreError("session turn does not belong to this task")
                receipt = commit_turn_commit(turn_commit, connection=conn)
                result.update(receipt["result_updates"])
                index_turn = (row["session_id"], turn_commit["record"])
            timestamp = time.time()
            conn.execute("""UPDATE tasks SET state=?,revision=revision+1,result_json=?,result_ref=?,
                failure_class=?,termination_reason=?,error_json=?,owner_instance_id=NULL,
                heartbeat_at=NULL,updated_at=?,finished_at=? WHERE request_id=?""",
                (resolved,_json_text(result),result_ref,failure_class,termination_reason,
                 _json_text(error or {}),timestamp,timestamp,request_id))
            updated = self._get(conn, request_id)
            self._event(conn, updated, "task_finalized", row["state"])
            task = self._task_row(updated)
        if index_turn is not None:
            update_case_index(*index_turn)
        return task

    def create_retry(self, parent_request_id, *, request_id, idempotency_key, turn_id=None, now=None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = self._get(conn, parent_request_id)
            if parent["state"] not in RETRYABLE_TASK_STATES:
                raise RetryNotAllowed(f"cannot retry task in {parent['state']}")
            attempt = parent["attempt_number"] + 1
            turn_id = turn_id or hashlib.sha256(
                f"{parent['turn_id']}:{idempotency_key}:{attempt}".encode()).hexdigest()[:32]
            return self._create(conn, request_id=request_id, idempotency_key=idempotency_key,
                session_id=parent["session_id"], turn_id=turn_id, request=_json_value(parent["request_json"]),
                parent_request_id=parent_request_id, attempt_number=attempt, now=now)

    def recover_startup(self):
        """A stopped process loses its running work; queued work remains runnable."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT * FROM tasks WHERE state IN ('running','cancelling')").fetchall()
            timestamp = time.time()
            for row in rows:
                conn.execute("""UPDATE tasks SET state='interrupted',revision=revision+1,
                    owner_instance_id=NULL,heartbeat_at=NULL,updated_at=?,finished_at=?,
                    failure_class='process_interrupted',termination_reason='service_restarted'
                    WHERE request_id=?""", (timestamp,timestamp,row["request_id"]))
                self._event(conn, self._get(conn,row["request_id"]), "startup_interrupted", row["state"])
            queued = [r[0] for r in conn.execute("SELECT request_id FROM tasks WHERE state='queued'")]
        return {"interrupted": [r["request_id"] for r in rows], "retained_queued": queued}

    def status(self):
        with self._connection() as conn:
            counts = dict(conn.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
            events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        return {"schema_version": TASK_STORE_SCHEMA_VERSION, "backend": "sqlite",
                "persistence_status": "durable", "task_count": sum(counts.values()),
                "state_counts": counts, "event_count": events}

    def close(self):
        with self._connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
