"""SQLite storage for session metadata and individually persisted turns."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterator


_SCHEMA_LOCK = threading.Lock()


class SQLiteSessionStore:
    def __init__(self, path: str | Path, *, connection: sqlite3.Connection | None = None) -> None:
        self.path = Path(path)
        self._active = connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._active is not None:
            yield self._active
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            with _SCHEMA_LOCK:
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(updated_at DESC);
                CREATE TABLE IF NOT EXISTS session_turns (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, turn_id)
                );
                CREATE INDEX IF NOT EXISTS idx_session_turns_order
                    ON session_turns(session_id, position);
                """)
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, session_id: str) -> Iterator[None]:
        if self._active is not None:
            yield
            return
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._active = connection
            try:
                yield
            finally:
                self._active = None

    def load(self, session_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            records = connection.execute(
                "SELECT record_json FROM session_turns WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return {
            **(json.loads(row[0]) if row else {}),
            "session_id": session_id,
            "turns": [json.loads(item[0]) for item in records],
        }

    def get_turn(self, session_id: str, turn_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM session_turns WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_turn(self, session_id: str, record: dict) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, updated_at) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
                (session_id, time.time_ns()),
            )
            connection.execute("""
                INSERT INTO session_turns(session_id, turn_id, position, record_json)
                VALUES (?, ?, (SELECT COALESCE(MAX(position), -1) + 1
                    FROM session_turns WHERE session_id = ?), ?)
                ON CONFLICT(session_id, turn_id) DO UPDATE SET record_json=excluded.record_json
            """, (session_id, record["turn_id"], session_id, json.dumps(record, ensure_ascii=False)))

    def delete_turn(self, session_id: str, turn_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM session_turns WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (time.time_ns(), session_id),
                )
            return cursor.rowcount

    def turn_count(self, session_id: str) -> int:
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM session_turns WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

    def available_ids(self) -> list[str]:
        with self._connection() as connection:
            return [row[0] for row in connection.execute("""
                SELECT session_id FROM sessions
                WHERE EXISTS (SELECT 1 FROM session_turns
                    WHERE session_turns.session_id = sessions.session_id)
                ORDER BY updated_at DESC, session_id
            """)]

    def import_session(self, session_id: str, state: dict, *, updated_at: int) -> bool:
        with self.transaction(session_id):
            with self._connection() as connection:
                metadata = {key: value for key, value in state.items()
                            if key not in {"session_id", "turns", "_persistence_status"}}
                cursor = connection.execute(
                    "INSERT INTO sessions(session_id, metadata_json, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO NOTHING",
                    (session_id, json.dumps(metadata, ensure_ascii=False), updated_at),
                )
                if not cursor.rowcount:
                    return False
                connection.executemany(
                    "INSERT INTO session_turns(session_id, turn_id, position, record_json) "
                    "VALUES (?, ?, ?, ?)",
                    [(session_id, turn["turn_id"], position, json.dumps(turn, ensure_ascii=False))
                     for position, turn in enumerate(state["turns"])],
                )
        return True
