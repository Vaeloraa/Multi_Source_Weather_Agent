from __future__ import annotations

"""Cross-session, evidence-only case memory.

This index stores bounded metadata projected from persisted evidence records.
It deliberately excludes answer prose and never performs implicit context
inheritance.  Callers must search it explicitly and retain the returned
session/turn/record references as provenance.
"""

from contextlib import contextmanager
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterable, Iterator, Mapping

from ..config import ROOT


CASE_MEMORY_SCHEMA_VERSION = "case-memory-v1"
CASE_MEMORY_SEARCH_SCHEMA_VERSION = "case-memory-search-v1"
DEFAULT_SESSION_ROOT = ROOT / "data" / "runtime" / "agent_sessions"
_SCHEMA_LOCK = threading.RLock()
_DATABASE_VERSION = 1
_EVENT_ID = re.compile(r"\b(?:R|S)?\d{10,20}\b", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z0-9_:+.-]+|[\u3400-\u9fff]")


def case_memory_path(session_root: str | Path | None = None) -> Path:
    configured = os.getenv("WEATHER_AGENT_CASE_MEMORY_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(session_root or os.getenv("WEATHER_AGENT_SESSION_DIR", "") or DEFAULT_SESSION_ROOT) / "_case_memory.sqlite3"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _walk(value: Any, *, depth: int = 0) -> Iterator[tuple[str, Any]]:
    if depth > 7:
        return
    if isinstance(value, Mapping):
        for key, item in islice(value.items(), 80):
            yield str(key), item
            yield from _walk(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in islice(value, 100):
            yield from _walk(item, depth=depth + 1)


def _strings(values: Any) -> list[str]:
    if isinstance(values, (list, tuple, set)):
        return [str(item).strip() for item in values if str(item).strip()]
    text = str(values or "").strip()
    return [text] if text else []


def _project_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a JSON evidence record whose region fields are scalar values."""
    request_facts = record.get("request_facts", {})
    calls = record.get("tool_evidence", [])
    sources = sorted({
        str(item.get("source") or "").strip().lower()
        for item in calls if str(item.get("source") or "").strip()
    })
    operations = sorted({
        f"{str(item.get('source') or '').strip().lower()}.{str(item.get('operation') or '').strip()}".strip(".")
        for item in calls if str(item.get("operation") or "").strip()
    })
    event_ids: set[str] = set()
    times: set[str] = set()
    models: set[str] = set()
    regions: list[dict[str, Any]] = []
    analogue_ids: set[str] = set()
    for key, item in _walk(record):
        lowered = key.lower()
        if lowered in {"event_id", "selected_event_id", "candidate_event_id"}:
            event_ids.update(value.upper() for value in _strings(item) if _EVENT_ID.fullmatch(value))
        if "time" in lowered and isinstance(item, str) and len(item) >= 10:
            times.add(item[:80])
        if lowered in {"model", "model_name", "model_id", "model_version"}:
            models.update(_strings(item))
        if lowered in {"analogue_event_id", "analog_event_id"}:
            analogue_ids.update(value.upper() for value in _strings(item))
        if lowered in {"region", "target_region", "aurora_region", "geo_context"} and isinstance(item, Mapping):
            region = {
                str(name): item[name]
                for name in ("south", "west", "north", "east", "latitude", "longitude", "query", "crs")
                if item.get(name) is not None
            }
            if region and region not in regions:
                regions.append(region)
    event_ids.update(match.group(0).upper() for match in _EVENT_ID.finditer(_json(request_facts)))
    models.update(source for source in sources if source in {"aurora", "cascast"})
    question = str(record.get("question") or "")[:1000]
    summary = {
        "record_id": str(record.get("record_id") or ""),
        "turn_id": str(record.get("turn_id") or ""),
        "created_at": str(record.get("created_at") or ""),
        "question": question,
        "status": str(record.get("status") or "unknown"),
        "event_ids": sorted(event_ids),
        "times": sorted(times)[:40],
        "regions": regions[:12],
        "sources": sources,
        "operations": operations,
        "models": sorted(models),
        "analogue_event_ids": sorted(analogue_ids),
        "artifact_count": len(record.get("artifacts", [])),
        "answer_prose_indexed": False,
    }
    return summary


class CaseMemoryStore:
    """SQLite WAL store for cross-session evidence metadata and typed edges."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else case_memory_path()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = NORMAL")
            if connection.execute("PRAGMA user_version").fetchone()[0] < _DATABASE_VERSION:
                with _SCHEMA_LOCK:
                    if connection.execute("PRAGMA user_version").fetchone()[0] < _DATABASE_VERSION:
                        self._initialize(connection)
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            f"""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS case_memory_records (
                    record_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    question TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_case_memory_session
                    ON case_memory_records(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_case_memory_case
                    ON case_memory_records(case_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_case_memory_created
                    ON case_memory_records(created_at DESC, record_id);
                CREATE INDEX IF NOT EXISTS idx_case_memory_turn
                    ON case_memory_records(session_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_case_memory_provenance
                    ON case_memory_records(json_extract(summary_json, '$.record_id'));
                CREATE TABLE IF NOT EXISTS case_memory_relations (
                    relation_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES case_memory_records(record_id)
                        ON DELETE CASCADE
                );
                DROP INDEX IF EXISTS idx_case_memory_relation_lookup;
                CREATE INDEX idx_case_memory_relation_lookup
                    ON case_memory_relations(predicate, object_type, object_id, record_id);
                CREATE INDEX IF NOT EXISTS idx_case_memory_relation_record
                    ON case_memory_relations(record_id, predicate);
                PRAGMA user_version = {_DATABASE_VERSION};
                COMMIT;
            """
        )

    @staticmethod
    def _relation(
        record_id: str,
        case_id: str,
        predicate: str,
        object_type: str,
        object_id: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, str, str, str, str, str, str]:
        identity = "\0".join((record_id, predicate, object_type, object_id))
        return (
            sha256(identity.encode("utf-8")).hexdigest(),
            record_id,
            "case",
            case_id,
            predicate,
            object_type,
            object_id,
            _json(dict(attributes or {})),
        )

    def upsert_turn(self, session_id: str, turn: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._upsert_turn(connection, session_id, turn)

    def _upsert_turn(
        self, connection: sqlite3.Connection, session_id: str, turn: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        record = turn.get("evidence_record")
        if not isinstance(record, Mapping):
            if turn.get("turn_id"):
                connection.execute(
                    "DELETE FROM case_memory_records WHERE session_id = ? AND turn_id = ?",
                    (str(session_id), str(turn["turn_id"])),
                )
            return None
        summary = _project_evidence(record)
        record_id = str(summary.get("record_id") or f"{session_id}:{summary.get('turn_id')}:evidence")
        turn_id = str(turn.get("turn_id") or summary.get("turn_id") or "")
        case_id = str(turn.get("study_case_id") or session_id)
        summary["record_id"] = record_id
        summary["turn_id"] = turn_id
        summary["session_id"] = str(session_id)
        summary["case_id"] = case_id
        record_id = f"{session_id}:{record_id}"
        searchable = " ".join(
            [summary["question"], case_id, *summary["event_ids"], *summary["times"],
             *summary["sources"], *summary["operations"], *summary["models"]]
        ).lower()
        relations = []
        for event_id in summary["event_ids"]:
            relations.append(self._relation(record_id, case_id, "has_event", "event", event_id))
        for value in summary["times"]:
            relations.append(self._relation(record_id, case_id, "has_valid_time", "time", value))
        for source in summary["sources"]:
            relations.append(self._relation(record_id, case_id, "used_source", "data_source", source))
        for operation in summary["operations"]:
            relations.append(self._relation(record_id, case_id, "performed", "operation", operation))
        for model in summary["models"]:
            relations.append(self._relation(record_id, case_id, "used_model", "model", model))
        for event_id in summary["analogue_event_ids"]:
            relations.append(self._relation(record_id, case_id, "retrieved_analogue", "event", event_id))
        for region in summary["regions"]:
            region_json = _json(region)
            region_id = sha256(region_json.encode("utf-8")).hexdigest()[:24]
            relations.append(self._relation(record_id, case_id, "has_geo_scope", "region", region_id, region))
        if any(token in operation.lower() for operation in summary["operations"] for token in ("verify", "evaluat", "metric", "score")):
            relations.append(self._relation(record_id, case_id, "has_evaluation", "evidence_role", "verification"))
        connection.execute(
            "DELETE FROM case_memory_records "
            "WHERE session_id = ? AND turn_id = ? AND record_id != ?",
            (str(session_id), turn_id, record_id),
        )
        connection.execute(
            """
            INSERT INTO case_memory_records(
                record_id, session_id, case_id, turn_id, created_at,
                status, question, summary_json, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                session_id=excluded.session_id,
                case_id=excluded.case_id,
                turn_id=excluded.turn_id,
                created_at=excluded.created_at,
                status=excluded.status,
                question=excluded.question,
                summary_json=excluded.summary_json,
                search_text=excluded.search_text,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                record_id, str(session_id), case_id, turn_id,
                str(summary.get("created_at") or turn.get("created_at") or ""),
                str(summary.get("status") or "unknown"), summary["question"],
                _json(summary), searchable,
            ),
        )
        connection.execute("DELETE FROM case_memory_relations WHERE record_id = ?", (record_id,))
        connection.executemany(
            """
            INSERT INTO case_memory_relations(
                relation_id, record_id, subject_type, subject_id,
                predicate, object_type, object_id, attributes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            relations,
        )
        return summary

    def delete_turn(self, session_id: str, turn_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM case_memory_records WHERE session_id = ? AND turn_id = ?",
                (str(session_id), str(turn_id)),
            )
        return int(cursor.rowcount)

    def relations(self, *, record_id: str | None = None, predicate: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if record_id is not None:
            clauses.append("record_id IN (SELECT record_id FROM case_memory_records "
                           "WHERE record_id = ? OR json_extract(summary_json, '$.record_id') = ?)")
            values.extend((str(record_id), str(record_id)))
        if predicate is not None:
            clauses.append("predicate = ?")
            values.append(str(predicate))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM case_memory_relations" + where + " ORDER BY predicate, object_id",
                values,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["attributes"] = json.loads(item.pop("attributes_json"))
            result.append(item)
        return result

    def search(
        self,
        query: str = "",
        *,
        event_id: str = "",
        source: str = "",
        operation: str = "",
        exclude_case_id: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        limit = max(1, min(50, int(limit)))
        query_tokens = sorted({item.lower() for item in _TOKEN.findall(str(query)) if item.strip()})
        normalized_event = str(event_id).strip().upper()
        normalized_source = str(source).strip().lower()
        normalized_operation = str(operation).strip().lower()
        clauses: list[str] = []
        values: list[Any] = []
        matched_fields = []
        base_score = 0.0
        for value, predicate, object_type, field, weight in (
            (normalized_event, "has_event", "event", "event_id", 4.0),
            (normalized_source, "used_source", "data_source", "source", 2.0),
            (normalized_operation, "performed", "operation", "operation", 2.0),
        ):
            if not value:
                continue
            condition = "instr(lower(object_id), ?) > 0" if field == "operation" else "object_id = ?"
            clauses.append(
                "r.record_id IN (SELECT record_id FROM case_memory_relations "
                f"WHERE predicate = ? AND object_type = ? AND {condition})"
            )
            values.extend((predicate, object_type, value))
            matched_fields.append(field)
            base_score += weight
        if exclude_case_id:
            clauses.append("r.case_id != ?")
            values.append(str(exclude_case_id))
        prefix = ""
        token_hits_sql = "0"
        if query_tokens:
            prefix = "WITH query_tokens AS (SELECT value FROM json_each(?)) "
            values.insert(0, _json(query_tokens))
            token_hits_sql = (
                "(SELECT COUNT(*) FROM query_tokens t WHERE instr(r.search_text, t.value) > 0)"
            )
            clauses.append("EXISTS (SELECT 1 FROM query_tokens t WHERE instr(r.search_text, t.value) > 0)")
            matched_fields.append("query_tokens")
        elif not matched_fields:
            base_score = 0.1
            matched_fields.append("recent")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        score_sql = f"{base_score} + 1.0 * {token_hits_sql} / {len(query_tokens) or 1}"
        order = "score DESC, r.created_at DESC, r.record_id" if query_tokens else "r.created_at DESC, r.record_id"
        with self._connection() as connection:
            connection.execute("BEGIN")
            total = connection.execute(
                prefix + "SELECT COUNT(*) FROM case_memory_records r" + where, values
            ).fetchone()[0]
            rows = connection.execute(
                prefix + f"SELECT r.*, {score_sql} AS score FROM case_memory_records r"
                + where + f" ORDER BY {order} LIMIT ?",
                [*values, limit],
            ).fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            summary = json.loads(row["summary_json"])
            matches.append({
                **summary,
                "score": float(row["score"]),
                "matched_fields": matched_fields.copy(),
                "matched_tokens": [token for token in query_tokens if token in row["search_text"]],
                "evidence_ref": {
                    "session_id": str(row["session_id"]),
                    "case_id": str(row["case_id"]),
                    "turn_id": str(row["turn_id"]),
                    "record_id": str(summary["record_id"]),
                },
            })
        return {
            "schema_version": CASE_MEMORY_SEARCH_SCHEMA_VERSION,
            "status": "ok",
            "query": {
                "text": str(query),
                "event_id": normalized_event,
                "source": normalized_source,
                "operation": normalized_operation,
                "excluded_case_id": str(exclude_case_id),
            },
            "matches": matches,
            "match_count": len(matches),
            "total_candidate_count": total,
            "answer_prose_used": False,
            "implicit_inheritance": False,
        }

    def status(self) -> dict[str, Any]:
        """Return bounded index facts without exposing its filesystem path."""

        with self._connection() as connection:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
            record_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM case_memory_records"
                ).fetchone()[0]
            )
            relation_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM case_memory_relations"
                ).fetchone()[0]
            )
            session_count = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM case_memory_records"
                ).fetchone()[0]
            )
            case_count = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT case_id) FROM case_memory_records"
                ).fetchone()[0]
            )
        return {
            "schema_version": CASE_MEMORY_SCHEMA_VERSION,
            "backend": "sqlite",
            "journal_mode": journal_mode,
            "record_count": record_count,
            "relation_count": relation_count,
            "session_count": session_count,
            "case_count": case_count,
            "answer_prose_indexed": False,
            "implicit_inheritance": False,
        }

    def rebuild(self, turns: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
        """Replace the complete derived index in one transaction from source turns."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM case_memory_records")
            indexed = sum(
                self._upsert_turn(connection, session_id, turn) is not None
                for session_id, turn in turns
            )
        return {
            "schema_version": CASE_MEMORY_SCHEMA_VERSION,
            "indexed_evidence_records": indexed,
        }


class CaseMemory:
    """Read-only facade injected into generated-code workers."""

    def __init__(self, store: CaseMemoryStore, *, current_case_id: str = "") -> None:
        self._store = store
        self.current_case_id = str(current_case_id)

    def search(
        self,
        query: str = "",
        *,
        event_id: str = "",
        source: str = "",
        operation: str = "",
        include_current_case: bool = False,
        limit: int = 5,
    ) -> dict[str, Any]:
        return self._store.search(
            query,
            event_id=event_id,
            source=source,
            operation=operation,
            exclude_case_id="" if include_current_case else self.current_case_id,
            limit=limit,
        )


__all__ = [
    "CASE_MEMORY_SCHEMA_VERSION",
    "CASE_MEMORY_SEARCH_SCHEMA_VERSION",
    "CaseMemory",
    "CaseMemoryStore",
    "case_memory_path",
]
